use anyhow::{anyhow, Ok, Result};
use std::collections::HashMap;
use tokenizers::Tokenizer;

pub struct SingleChineseTokenizer {
    tokenizer: Tokenizer,
    split_map: HashMap<u32, Vec<u32>>,
}

impl SingleChineseTokenizer {
    pub fn new(path: &str) -> Result<Self> {
        let path = path.to_string();
        assert!(
            std::path::Path::new(&path).exists(),
            "model path file not exists"
        );
        let tokenizer_file = path.clone() + "/tokenizer.json";
        assert!(
            std::path::Path::new(&tokenizer_file).exists(),
            "tokenizer. json not exists in model path"
        );
        let tokenizer = Tokenizer::from_file(tokenizer_file)
            .map_err(|e| anyhow!(format!("tokenizer from file error{e}")))?;
        let mut split_map = HashMap::new();
        for (token, token_id) in tokenizer.get_vocab(false) {
            let clean_token = token.replace("▁", "");
            let mut chars = clean_token.chars();
            let is_multichar_cjk =
                chars.clone().count() >= 2 && chars.all(Self::is_cjk);
            if is_multichar_cjk {
                let char_ids = clean_token
                    .chars()
                    .map(|c| tokenizer.token_to_id(&c.to_string()))
                    .collect::<Option<Vec<_>>>();
                if let Some(char_ids) = char_ids {
                    split_map.insert(token_id, char_ids);
                }
            }
        }
        Ok(Self {
            tokenizer,
            split_map,
        })
    }

    pub fn encode(&self, text: String) -> Result<Vec<u32>> {
        let encode = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!(format!("tokenizer encode error:  {e}")))?;
        let mut ids = Vec::new();
        for id in encode.get_ids() {
            if let Some(expanded) = self.split_map.get(id) {
                ids.extend_from_slice(expanded);
            } else {
                ids.push(*id);
            }
        }
        Ok(ids)
    }

    fn is_cjk(c: char) -> bool {
        matches!(
            c as u32,
            0x4E00..=0x9FFF
                | 0x3400..=0x4DBF
                | 0xF900..=0xFAFF
                | 0x20000..=0x2A6DF
        )
    }
}
