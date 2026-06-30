use anyhow::{Ok, Result, anyhow};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Embedding, Module, VarBuilder, embedding};

use crate::utils::bucketize;

pub struct CausalConv1d {
    conv1d: Conv1d,
    padding: usize,
}

impl CausalConv1d {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        padding: usize,
        dilation: usize,
        groups: usize,
        stride: usize,
    ) -> Result<Self> {
        let config = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };

        let conv1d = Conv1d::new(weight, bias, config);
        Ok(Self { conv1d, padding })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_pad = x
            .pad_with_zeros(D::Minus1, self.padding * 2, 0)?
            .contiguous()?;
        let x = self.conv1d.forward(&x_pad)?;
        Ok(x)
    }

    /// Note: a reusable `[state|x]` pad buffer would need true in-place writes; Candle
    /// `slice_assign` returns a fresh tensor each time, so `Tensor::cat` is kept here.
    pub fn forward_stream(&self, x: &Tensor, state: &mut Tensor) -> Result<Tensor> {
        // x: [B, C, L]
        // state: [B, C, P*2]
        let state_len = state.dim(D::Minus1)?;
        let x_pad = if state_len == 0 {
            x.clone()
        } else {
            Tensor::cat(&[state as &Tensor, x], D::Minus1)?
        };
        let x_out = self.conv1d.forward(&x_pad)?;
        *state = x_pad.narrow(D::Minus1, x.dim(D::Minus1)?, state_len)?;
        Ok(x_out)
    }

    pub fn init_state(&self, batch_size: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let in_channels = self.conv1d.weight().dim(1)?;
        let groups = self.conv1d.config().groups;
        let t = Tensor::zeros(
            (batch_size, in_channels * groups, self.padding * 2),
            dtype,
            device,
        )?;
        Ok(t)
    }
}

pub struct CausalConvTranspose1d {
    conv_transpose1d: ConvTranspose1d,
    padding: usize,
    output_padding: usize,
}

// 元素间：stride-1
// 两边： k-p-1
// (h-1)*s -2p+k
// (h+1)*s

impl CausalConvTranspose1d {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        padding: usize,
        dilation: usize,
        output_padding: usize,
        groups: usize,
        stride: usize,
    ) -> Result<Self> {
        let config = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation,
            groups,
        };

        let conv_transpose1d = ConvTranspose1d::new(weight, bias, config);
        Ok(Self {
            conv_transpose1d,
            padding,
            output_padding,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv_transpose1d.forward(&x.contiguous()?)?;
        let last_dim = x.dim(D::Minus1)?;
        let select_num = last_dim.saturating_sub(self.padding * 2 - self.output_padding);
        let x = x.narrow(D::Minus1, 0, select_num)?;
        Ok(x)
    }

    pub fn forward_stream(&self, x: &Tensor, state: &mut Tensor) -> Result<Tensor> {
        // x: [B, C, L]
        // state: [B, C, K-S] overlap buffer
        let x_out = self.conv_transpose1d.forward(x)?;
        let kernel_size = self.conv_transpose1d.weight().dim(2)?;
        let stride = self.conv_transpose1d.config().stride;
        let overlap_len = kernel_size - stride;
        let out_len = x_out.dim(D::Minus1)?;
        let select_num = out_len.saturating_sub(overlap_len);

        let mut out = x_out.narrow(D::Minus1, 0, select_num)?;
        if overlap_len > 0 {
            // Fold overlap add + tail into one elementwise add: state padded with zeros on the
            // right so only the first min(overlap_len, select_num) positions contribute, matching
            // the old narrow + add + cat path without an extra cat kernel.
            let state_contrib = if select_num <= overlap_len {
                state.narrow(D::Minus1, 0, select_num)?
            } else {
                state.pad_with_zeros(D::Minus1, 0, select_num - overlap_len)?
            };
            out = out.add(&state_contrib)?;
        }
        *state = x_out.narrow(D::Minus1, select_num, out_len - select_num)?;
        Ok(out)
    }

    pub fn init_state(&self, batch_size: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let out_channels = self.conv_transpose1d.weight().dim(1)?;
        let kernel_size = self.conv_transpose1d.weight().dim(2)?;
        let stride = self.conv_transpose1d.config().stride;
        let t = Tensor::zeros(
            (batch_size, out_channels, kernel_size - stride),
            dtype,
            device,
        )?;
        Ok(t)
    }
}

pub struct WNCausalConv1d {
    conv: CausalConv1d,
}
impl WNCausalConv1d {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        stride: usize,
    ) -> Result<Self> {
        let in_c = in_c / groups;
        let weight_g = vb.get((out_c, 1, 1), "weight_g")?;
        let weight_v = vb.get((out_c, in_c, kernel_size), "weight_v")?;
        let bias = vb.get(out_c, "bias").ok();
        let weight_norm = weight_v.sqr()?.sum_keepdim(1)?.sum_keepdim(2)?.sqrt()?;
        let normalized_weight = weight_v.broadcast_div(&weight_norm)?;
        let scaled_weight = normalized_weight.broadcast_mul(&weight_g)?;
        let conv = CausalConv1d::new(scaled_weight, bias, padding, dilation, groups, stride)?;
        Ok(Self { conv })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        Ok(x)
    }
    pub fn forward_stream(&self, x: &Tensor, state: &mut Tensor) -> Result<Tensor> {
        self.conv.forward_stream(x, state)
    }
    pub fn init_state(&self, batch_size: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        self.conv.init_state(batch_size, device, dtype)
    }
}

pub struct WNCausalConvTranspose1d {
    conv_transpose: CausalConvTranspose1d,
}

impl WNCausalConvTranspose1d {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        dilation: usize,
        kernel_size: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
        stride: usize,
    ) -> Result<Self> {
        let in_c = in_c / groups;
        let weight_g = vb.get((in_c, 1, 1), "weight_g")?;
        let weight_v = vb.get((in_c, out_c, kernel_size), "weight_v")?;
        let bias = vb.get(out_c, "bias").ok();
        let weight_norm = weight_v.sqr()?.sum_keepdim(1)?.sum_keepdim(2)?.sqrt()?;
        let normalized_weight = weight_v.broadcast_div(&weight_norm)?;
        let scaled_weight = normalized_weight.broadcast_mul(&weight_g)?;
        let conv_transpose = CausalConvTranspose1d::new(
            scaled_weight,
            bias,
            padding,
            dilation,
            output_padding,
            groups,
            stride,
        )?;
        Ok(Self { conv_transpose })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv_transpose.forward(x)?;
        Ok(x)
    }
    pub fn forward_stream(&self, x: &Tensor, state: &mut Tensor) -> Result<Tensor> {
        self.conv_transpose.forward_stream(x, state)
    }
    pub fn init_state(&self, batch_size: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        self.conv_transpose.init_state(batch_size, device, dtype)
    }
}

pub struct Snake1d {
    alpha: Tensor,
    alpha_recip: Tensor,
}
impl Snake1d {
    pub fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        let alpha_recip = alpha.affine(1.0, 1e-9)?.recip()?;
        Ok(Self { alpha, alpha_recip })
    }

    // x + sin(alpha*x)^2 / alpha
    #[inline]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ax = x.broadcast_mul(&self.alpha)?;
        let x_out = ax.sin()?.sqr()?.broadcast_mul(&self.alpha_recip)?;
        Ok(x.add(&x_out)?)
    }
}

pub struct CausalResidualUnit {
    // pad: usize,
    block0: Snake1d,
    block1: WNCausalConv1d,
    block2: Snake1d,
    block3: WNCausalConv1d,
}

impl CausalResidualUnit {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        dilation: usize,
        kernel: usize,
        groups: usize,
    ) -> Result<Self> {
        let pad = ((kernel - 1) * dilation) / 2;
        let block0 = Snake1d::new(vb.pp("block.0"), dim)?;
        let block1 =
            WNCausalConv1d::new(vb.pp("block.1"), dim, dim, kernel, dilation, pad, groups, 1)?;
        let block2 = Snake1d::new(vb.pp("block.2"), dim)?;
        let block3 = WNCausalConv1d::new(vb.pp("block.3"), dim, dim, 1, 1, 0, 1, 1)?;
        Ok(Self {
            // pad,
            block0,
            block1,
            block2,
            block3,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // let orig_dim = x.dims();
        let last_dim_x = x.dim(D::Minus1)?;
        let y = self.block0.forward(x)?;
        let y = self.block1.forward(&y)?;
        let y = self.block2.forward(&y)?;
        let y = self.block3.forward(&y)?;
        // let dim = y.dims();
        let last_dim_y = y.dim(D::Minus1)?;
        let pad = (last_dim_x - last_dim_y) / 2;
        if pad > 0 {
            let res_x = x.narrow(D::Minus1, pad, last_dim_y)?;
            Ok(y.add(&res_x)?)
        } else {
            Ok(y.add(x)?)
        }
    }

    pub fn forward_stream(&self, x: &Tensor, states: &mut Vec<Tensor>) -> Result<Tensor> {
        let y = self.block0.forward(x)?;
        let y = self.block1.forward_stream(&y, &mut states[0])?;
        let y = self.block2.forward(&y)?;
        let y = self.block3.forward_stream(&y, &mut states[1])?;
        Ok(y.add(x)?)
    }

    pub fn init_states(
        &self,
        batch_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Vec<Tensor>> {
        Ok(vec![
            self.block1.init_state(batch_size, device, dtype)?,
            self.block3.init_state(batch_size, device, dtype)?,
        ])
    }
}

pub struct CausalEncoderBlock {
    block0: CausalResidualUnit,
    block1: CausalResidualUnit,
    block2: CausalResidualUnit,
    block3: Snake1d,
    block4: WNCausalConv1d,
}

impl CausalEncoderBlock {
    pub fn new(
        vb: VarBuilder,
        in_dim: Option<usize>,
        out_dim: usize,
        stride: usize,
        groups: usize,
    ) -> Result<Self> {
        let in_dim = match in_dim {
            Some(d) => d,
            None => out_dim / 2,
        };
        let block0 = CausalResidualUnit::new(vb.pp("block.0"), in_dim, 1, 7, groups)?;
        let block1 = CausalResidualUnit::new(vb.pp("block.1"), in_dim, 3, 7, groups)?;
        let block2 = CausalResidualUnit::new(vb.pp("block.2"), in_dim, 9, 7, groups)?;
        let block3 = Snake1d::new(vb.pp("block.3"), in_dim)?;
        let padding = (stride as f32 / 2.0).ceil() as usize;
        let block4 = WNCausalConv1d::new(
            vb.pp("block.4"),
            in_dim,
            out_dim,
            2 * stride,
            1,
            padding,
            1,
            stride,
        )?;
        Ok(Self {
            block0,
            block1,
            block2,
            block3,
            block4,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.block0.forward(x)?;
        let x = self.block1.forward(&x)?;
        let x = self.block2.forward(&x)?;
        let x = self.block3.forward(&x)?;
        let x = self.block4.forward(&x)?;
        Ok(x)
    }
}

pub struct CausalEncoder {
    block0: WNCausalConv1d,
    blocks: Vec<CausalEncoderBlock>,
    fc_mu: WNCausalConv1d,
    fc_logvar: WNCausalConv1d,
}

impl CausalEncoder {
    pub fn new(
        vb: VarBuilder,
        d_model: usize,
        laten_dim: usize,
        strides: Vec<usize>,
        depthwise: bool,
    ) -> Result<Self> {
        let mut d_model = d_model;
        let mut groups;
        let block0 = WNCausalConv1d::new(vb.pp("block.0"), 1, d_model, 7, 1, 3, 1, 1)?;
        let vb_block = vb.pp("block");
        let mut blocks = Vec::new();
        for (i, stride) in strides.iter().enumerate() {
            d_model *= 2;
            groups = if depthwise { d_model / 2 } else { 1 };
            let block_i =
                CausalEncoderBlock::new(vb_block.pp(i + 1), None, d_model, *stride, groups)?;
            blocks.push(block_i);
        }
        let fc_mu = WNCausalConv1d::new(vb.pp("fc_mu"), d_model, laten_dim, 3, 1, 1, 1, 1)?;
        let fc_logvar = WNCausalConv1d::new(vb.pp("fc_logvar"), d_model, laten_dim, 3, 1, 1, 1, 1)?;
        Ok(Self {
            block0,
            blocks,
            fc_mu,
            fc_logvar,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let x = x.contiguous()?;
        let mut hidden_state = self.block0.forward(&x)?;
        for block_i in &self.blocks {
            hidden_state = block_i.forward(&hidden_state)?;
        }
        let mu = self.fc_mu.forward(&hidden_state)?;
        let logvar = self.fc_logvar.forward(&hidden_state)?;
        Ok((hidden_state, mu, logvar))
    }
}

pub struct NoiseBlock {
    linear: WNCausalConv1d,
}

impl NoiseBlock {
    pub fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let linear = WNCausalConv1d::new(vb.pp("linear"), dim, dim, 1, 1, 0, 1, 1)?;
        Ok(Self { linear })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (bs, _, t) = x.dims3()?;
        let noise = Tensor::randn(0.0_f32, 1.0, (bs, 1, t), x.device())?.to_dtype(x.dtype())?;
        let h = self.linear.forward(x)?;
        let n = h.broadcast_mul(&noise)?;
        let x = x.add(&n)?;
        Ok(x)
    }
}

pub struct CausalDecoderBlock {
    block0: Snake1d,
    block1: WNCausalConvTranspose1d,
    block2: CausalResidualUnit,
    block3: CausalResidualUnit,
    block4: CausalResidualUnit,
}

impl CausalDecoderBlock {
    pub fn new(
        vb: VarBuilder,
        input_dim: usize,
        output_dim: usize,
        stride: usize,
        groups: usize,
    ) -> Result<Self> {
        let block0 = Snake1d::new(vb.pp("block.0"), input_dim)?;
        let padding = (stride as f32 / 2.0).ceil() as usize;
        let block1 = WNCausalConvTranspose1d::new(
            vb.pp("block.1"),
            input_dim,
            output_dim,
            1,
            2 * stride,
            padding,
            stride % 2,
            1,
            stride,
        )?;
        let block2 = CausalResidualUnit::new(vb.pp("block.2"), output_dim, 1, 7, groups)?;
        let block3 = CausalResidualUnit::new(vb.pp("block.3"), output_dim, 3, 7, groups)?;
        let block4 = CausalResidualUnit::new(vb.pp("block.4"), output_dim, 9, 7, groups)?;
        Ok(Self {
            block0,
            block1,
            block2,
            block3,
            block4,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.block0.forward(x)?;
        let x = self.block1.forward(&x)?;
        let x = self.block2.forward(&x)?;
        let x = self.block3.forward(&x)?;
        let x = self.block4.forward(&x)?;
        Ok(x)
    }

    pub fn forward_stream(&self, x: &Tensor, states: &mut DecoderBlockState) -> Result<Tensor> {
        let x = self.block0.forward(x)?;
        let x = self.block1.forward_stream(&x, &mut states[0][0])?;
        let x = self.block2.forward_stream(&x, &mut states[1])?;
        let x = self.block3.forward_stream(&x, &mut states[2])?;
        let x = self.block4.forward_stream(&x, &mut states[3])?;
        Ok(x)
    }

    pub fn init_states(
        &self,
        batch_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<DecoderBlockState> {
        Ok(vec![
            vec![self.block1.init_state(batch_size, device, dtype)?],
            self.block2.init_states(batch_size, device, dtype)?,
            self.block3.init_states(batch_size, device, dtype)?,
            self.block4.init_states(batch_size, device, dtype)?,
        ])
    }
}

pub struct SampleRateConditionLayer {
    cond_type: String,
    scale_embed: Option<Embedding>,
    bias_embed: Option<Embedding>,
    cond_embed: Option<Embedding>,
}

impl SampleRateConditionLayer {
    pub fn new(
        vb: VarBuilder,
        input_dim: usize,
        sr_bin_buckets_len: usize,
        cond_type: String,
    ) -> Result<Self> {
        let (scale_embed, bias_embed, cond_embed) = if cond_type.contains("scale_bias") {
            let scale_embed = embedding(sr_bin_buckets_len, input_dim, vb.pp("scale_embed"))?;
            let bias_embed = embedding(sr_bin_buckets_len, input_dim, vb.pp("bias_embed"))?;
            (Some(scale_embed), Some(bias_embed), None)
        } else if cond_type.eq("add") {
            let cond_embed = embedding(sr_bin_buckets_len, input_dim, vb.pp("cond_embed"))?;
            (None, None, Some(cond_embed))
        } else {
            (None, None, None)
        };
        Ok(Self {
            cond_type,
            scale_embed,
            bias_embed,
            cond_embed,
        })
    }

    pub fn forward(&self, x: &Tensor, sr_cond: &Tensor) -> Result<Tensor> {
        if self.cond_type.contains("scale_bias") {
            if let Some(scale_embed) = &self.scale_embed {
                if let Some(bias_embed) = &self.bias_embed {
                    return Ok(x
                        .broadcast_mul(&scale_embed.forward(sr_cond)?.unsqueeze(D::Minus1)?)?
                        .broadcast_add(&bias_embed.forward(sr_cond)?.unsqueeze(D::Minus1)?)?);
                }
            }
        } else if self.cond_type.eq("add") {
            if let Some(cond_embed) = &self.cond_embed {
                return Ok(x.broadcast_add(&cond_embed.forward(sr_cond)?.unsqueeze(D::Minus1)?)?);
            }
        }
        Err(anyhow!("unsupported cond_type: {}", self.cond_type))
    }
}

pub struct CausalDecoder {
    model0: WNCausalConv1d,
    model1: WNCausalConv1d,
    models: Vec<CausalDecoderBlock>,
    model_minus_2: Snake1d,
    model_minus_1: WNCausalConv1d,
    sr_bin_boundaries: Option<Vec<usize>>,
    sr_cond_model: Option<Vec<SampleRateConditionLayer>>,
}

impl CausalDecoder {
    pub fn new(
        vb: VarBuilder,
        input_channel: usize,
        channels: usize,
        rates: Vec<usize>,
        d_out: usize,
        depthwise: bool,
        sr_bin_boundaries: Option<Vec<usize>>,
        cond_type: Option<String>,
    ) -> Result<Self> {
        let model0 = WNCausalConv1d::new(
            vb.pp("model.0"),
            input_channel,
            input_channel,
            7,
            1,
            3,
            input_channel,
            1,
        )?;
        let model1 = WNCausalConv1d::new(vb.pp("model.1"), input_channel, channels, 1, 1, 0, 1, 1)?;
        let vb_model = vb.pp("model");
        let mut output_dim = channels;
        let mut models = Vec::new();
        let mut input_channels_vec = vec![];
        for (i, stride) in rates.iter().enumerate() {
            let input_dim = channels / 2_usize.pow(i as u32);
            input_channels_vec.push(input_dim);
            output_dim = channels / 2_usize.pow((i + 1) as u32);
            let groups = if depthwise { output_dim } else { 1 };
            let model_i = CausalDecoderBlock::new(
                vb_model.pp(i + 2),
                input_dim,
                output_dim,
                *stride,
                groups,
            )?;
            models.push(model_i);
        }
        let idx = rates.len() + 2;
        let model_minus_2 = Snake1d::new(vb_model.pp(idx), output_dim)?;
        let model_minus_1 =
            WNCausalConv1d::new(vb_model.pp(idx + 1), output_dim, d_out, 7, 1, 3, 1, 1)?;
        let (sr_cond_model, sr_bin_boundaries) =
            if let Some(sr) = sr_bin_boundaries.clone() {
                if let Some(cond_type) = cond_type {
                    let sr_len = sr.len() + 1;
                    let vb_sr = vb.pp("sr_cond_model");
                    let mut sr_cond_model = vec![];
                    for (i, &input_dim) in input_channels_vec.iter().enumerate() {
                        let layer = SampleRateConditionLayer::new(
                            vb_sr.pp(i + 2),
                            input_dim,
                            sr_len,
                            cond_type.clone(),
                        )?;
                        sr_cond_model.push(layer);
                    }
                    (Some(sr_cond_model), Some(sr))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };
        Ok(Self {
            model0,
            model1,
            models,
            model_minus_2,
            model_minus_1,
            sr_bin_boundaries,
            sr_cond_model,
        })
    }

    fn sr_cond_tensor(&self, sr_cond: Option<usize>, device: &Device) -> Result<Option<Tensor>> {
        let Some(sr_cond) = sr_cond else {
            return Ok(None);
        };
        let Some(boundaries) = &self.sr_bin_boundaries else {
            return Ok(None);
        };
        let sr = bucketize(sr_cond, boundaries)?;
        Ok(Some(Tensor::new(vec![sr as u32], device)?))
    }

    pub fn forward(&self, x: &Tensor, sr_cond: Option<usize>) -> Result<Tensor> {
        let x = self.model0.forward(x)?;
        let mut x = self.model1.forward(&x)?;
        if let Some(sr_cond_tensor) = self.sr_cond_tensor(sr_cond, x.device())? {
            if let Some(sr_models) = &self.sr_cond_model {
                for (model_i, sr_model_i) in self.models.iter().zip(sr_models.iter()) {
                    x = sr_model_i.forward(&x, &sr_cond_tensor)?;
                    x = model_i.forward(&x)?;
                }
            } else {
                for model_i in &self.models {
                    x = model_i.forward(&x)?;
                }
            }
        } else {
            for model_i in &self.models {
                x = model_i.forward(&x)?;
            }
        }
        let x = self.model_minus_2.forward(&x)?;
        let x = self.model_minus_1.forward(&x)?;
        Ok(x.tanh()?)
    }

    pub fn forward_stream(
        &self,
        x: &Tensor,
        state: &mut DecoderState,
        sr_cond: Option<usize>,
    ) -> Result<Tensor> {
        let mut x = self.model0.forward_stream(x, &mut state.model0_state)?;
        x = self.model1.forward_stream(&x, &mut state.model1_state)?;
        if let Some(sr_cond_tensor) = self.sr_cond_tensor(sr_cond, x.device())? {
            if let Some(sr_models) = &self.sr_cond_model {
                for (i, model_i) in self.models.iter().enumerate() {
                    x = sr_models[i].forward(&x, &sr_cond_tensor)?;
                    x = model_i.forward_stream(&x, &mut state.block_states[i])?;
                }
            } else {
                for (i, model_i) in self.models.iter().enumerate() {
                    x = model_i.forward_stream(&x, &mut state.block_states[i])?;
                }
            }
        } else {
            for (i, model_i) in self.models.iter().enumerate() {
                x = model_i.forward_stream(&x, &mut state.block_states[i])?;
            }
        }
        x = self.model_minus_2.forward(&x)?;
        x = self
            .model_minus_1
            .forward_stream(&x, &mut state.model_minus_1_state)?;
        x = x.tanh()?;
        Ok(x)
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<DecoderState> {
        let mut block_states = Vec::new();
        for model_i in &self.models {
            block_states.push(model_i.init_states(batch_size, device, dtype)?);
        }
        Ok(DecoderState {
            model0_state: self.model0.init_state(batch_size, device, dtype)?,
            model1_state: self.model1.init_state(batch_size, device, dtype)?,
            block_states,
            model_minus_1_state: self.model_minus_1.init_state(batch_size, device, dtype)?,
        })
    }
}

pub struct DecoderState {
    pub model0_state: Tensor,
    pub model1_state: Tensor,
    pub block_states: Vec<DecoderBlockState>,
    pub model_minus_1_state: Tensor,
}

pub type DecoderBlockState = Vec<Vec<Tensor>>; // [block1, block2, block3] states

pub struct AudioVAE {
    pub latent_dim: usize,
    hop_length: usize,
    encoder: CausalEncoder,
    decoder: CausalDecoder,
    dtype: DType,
    pub sample_rate: usize,
    pub out_sample_rate: usize,
    pub chunk_size: usize,
}

impl AudioVAE {
    #[inline]
    fn to_compute_dtype(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.dtype() == self.dtype {
            Ok(tensor.clone())
        } else {
            Ok(tensor.to_dtype(self.dtype)?)
        }
    }

    #[inline]
    fn to_output_dtype(&self, tensor: Tensor) -> Result<Tensor> {
        if tensor.dtype() == DType::F32 {
            Ok(tensor)
        } else {
            Ok(tensor.to_dtype(DType::F32)?)
        }
    }

    pub fn new(
        vb: VarBuilder,
        encoder_dim: usize,
        encoder_rates: Vec<usize>,
        laten_dim: Option<usize>,
        decoder_dim: usize,
        decoder_rates: Vec<usize>,
        sample_rate: usize,
        out_sample_rate: usize,
        sr_bin_boundaries: Option<Vec<usize>>,
        cond_type: Option<String>,
    ) -> Result<Self> {
        let dtype = vb.dtype();
        let latent_dim = match laten_dim {
            Some(d) => d,
            None => encoder_dim * (2_usize.pow(encoder_rates.len() as u32)),
        };
        let hop_length = encoder_rates.iter().product();
        let encoder = CausalEncoder::new(
            vb.pp("encoder"),
            encoder_dim,
            latent_dim,
            encoder_rates.clone(),
            true,
        )?;
        let decoder = CausalDecoder::new(
            vb.pp("decoder"),
            latent_dim,
            decoder_dim,
            decoder_rates.clone(),
            1,
            true,
            sr_bin_boundaries,
            cond_type,
        )?;
        let chunk_size = hop_length;
        Ok(Self {
            latent_dim,
            hop_length,
            encoder,
            decoder,
            dtype,
            sample_rate,
            out_sample_rate,
            chunk_size,
        })
    }

    pub fn preprocess(&self, audio_data: &Tensor, sample_rate: Option<usize>) -> Result<Tensor> {
        let sample_rate = match sample_rate {
            Some(r) => r,
            None => self.sample_rate,
        };
        assert_eq!(sample_rate, self.sample_rate);
        let pad_to = self.hop_length;
        let length = audio_data.dim(D::Minus1)?;
        let right_pad = (length as f32 / pad_to as f32).ceil() as usize * pad_to - length;
        let audio_data = audio_data.pad_with_zeros(D::Minus1, 0, right_pad)?;
        Ok(audio_data)
    }

    pub fn decode(&self, z: &Tensor, sr_cond: Option<usize>) -> Result<Tensor> {
        let z = self.to_compute_dtype(z)?;
        let sr_cond = if sr_cond.is_none() && self.decoder.sr_bin_boundaries.is_some() {
            Some(self.out_sample_rate)
        } else {
            sr_cond
        };
        let x = self.decoder.forward(&z, sr_cond)?;
        self.to_output_dtype(x)
    }

    pub fn decode_stream(
        &self,
        z: &Tensor,
        state: &mut DecoderState,
        sr_cond: Option<usize>,
    ) -> Result<Tensor> {
        let z = self.to_compute_dtype(z)?;
        let sr_cond = if sr_cond.is_none() && self.decoder.sr_bin_boundaries.is_some() {
            Some(self.out_sample_rate)
        } else {
            sr_cond
        };
        let x = self.decoder.forward_stream(&z, state, sr_cond)?;
        self.to_output_dtype(x)
    }

    pub fn init_decoder_state(
        &self,
        batch_size: usize,
        device: &Device,
        _dtype: DType,
    ) -> Result<DecoderState> {
        self.decoder.init_state(batch_size, device, self.dtype)
    }

    #[must_use]
    pub fn compute_dtype(&self) -> DType {
        self.dtype
    }

    pub fn encode(&self, audio_data: &Tensor, sample_rate: Option<usize>) -> Result<Tensor> {
        let audio_data = match audio_data.rank() {
            2 => audio_data.unsqueeze(1)?,
            _ => audio_data.clone(),
        };
        let audio_data = self.to_compute_dtype(&self.preprocess(&audio_data, sample_rate)?)?;
        let (_, mu, _) = self.encoder.forward(&audio_data)?;
        Ok(mu)
    }
}

#[cfg(test)]
mod streaming_overlap_tests {
    use super::*;
    use candle_core::Device;

    fn legacy_overlap_merge(out: &Tensor, state: &Tensor, overlap_len: usize) -> Result<Tensor> {
        let select_num = out.dim(D::Minus1)?;
        if overlap_len == 0 {
            return Ok(out.clone());
        }
        let add_len = overlap_len.min(select_num);
        let a = out.narrow(D::Minus1, 0, add_len)?;
        let b = state.narrow(D::Minus1, 0, add_len)?;
        let sum = a.add(&b)?;
        if select_num > add_len {
            Ok(Tensor::cat(
                &[&sum, &out.narrow(D::Minus1, add_len, select_num - add_len)?],
                D::Minus1,
            )?)
        } else {
            Ok(sum)
        }
    }

    fn merged_overlap_add(out: &Tensor, state: &Tensor, overlap_len: usize) -> Result<Tensor> {
        let select_num = out.dim(D::Minus1)?;
        if overlap_len == 0 {
            return Ok(out.clone());
        }
        let state_contrib = if select_num <= overlap_len {
            state.narrow(D::Minus1, 0, select_num)?
        } else {
            state.pad_with_zeros(D::Minus1, 0, select_num - overlap_len)?
        };
        Ok(out.add(&state_contrib)?)
    }

    #[test]
    fn causal_transpose_overlap_matches_legacy_cat() -> Result<()> {
        let device = Device::Cpu;
        for overlap_len in [1usize, 3, 7, 15] {
            for select_num in 1usize..=24 {
                let out = Tensor::randn(0f32, 1f32, (2, 4, select_num), &device)?;
                let state = Tensor::randn(0f32, 1f32, (2, 4, overlap_len), &device)?;
                let legacy = legacy_overlap_merge(&out, &state, overlap_len)?;
                let merged = merged_overlap_add(&out, &state, overlap_len)?;
                let max_abs = legacy.sub(&merged)?.abs()?.max_all()?.to_scalar::<f32>()?;
                assert!(
                    max_abs < 1e-5,
                    "overlap_len={overlap_len} select_num={select_num} max_abs={max_abs}"
                );
            }
        }
        Ok(())
    }
}
