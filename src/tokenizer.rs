use anyhow::{anyhow, Ok, Result};
use tokenizers::Tokenizer;

pub struct SingleChineseTokenizer {
    tokenizer: Tokenizer,
    multichar_tokens: Vec<String>,
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
        let mut multichar_tokens = Vec::new();
        for (token, _) in tokenizer.get_vocab(false) {
            let len = token.chars().count();
            if len >= 2 {
                let is_chinese = token.chars().all(|c| {
                    let c_ = c as u32;
                    (0x4E00..=0x9FFF).contains(&c_)
                });
                if is_chinese {
                    multichar_tokens.push(token);
                }
            }
        }
        Ok(Self {
            tokenizer,
            multichar_tokens,
        })
    }

    pub fn encode(&self, text: String) -> Result<Vec<u32>> {
        let encode = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!(format!("tokenizer encode error:  {e}")))?;
        let tokens = encode.get_tokens();
        let mut split_character = Vec::new();
        for token in tokens {
            let clean_token = token.replace("▁", "");
            if self.multichar_tokens.contains(&clean_token) {
                let chars: Vec<String> = clean_token.chars().map(|c| c.to_string()).collect();
                split_character.extend(chars);
            } else {
                split_character.push(token.clone());
            }
        }
        let ids: Vec<u32> = split_character
            .iter()
            .filter_map(|c| self.tokenizer.token_to_id(c))
            .collect();
        Ok(ids)
    }
}
