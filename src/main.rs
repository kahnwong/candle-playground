mod token_output_stream;
use anyhow::{Error as E, Result};
use clap::Parser;
mod candle_server_utils;
use candle_transformers::models::gemma::{Config as Config1, Model as Model1};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use token_output_stream::TokenOutputStream;
use tokenizers::Tokenizer;


struct TextGeneration {
    model: Model1,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model1,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<eos>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <eos> token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    prompt: String,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse(); // [TODO] remove

    // log accelerators
    println!(
        "cuda: {}, metal: {}",
        candle_core::utils::cuda_is_available(),
        candle_core::utils::metal_is_available(),
    );

    // log parameters
    const TEMPERATURE: Option<f64> = Some(0.0);
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    const REPEAT_PENALTY: f32 = 1.1;
    /// The context size to consider for the repeat penalty.
    const REPEAT_LAST_N: usize = 64;

    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        TEMPERATURE.unwrap(),
        REPEAT_PENALTY,
        REPEAT_LAST_N
    );

    // ------------ get assets ------------
    let start = std::time::Instant::now();

    let api = Api::new()?;
    let model_id = "google/gemma-1.1-2b-it".to_string();
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        "main".to_string(),
    ));

    let tokenizer_filename = repo.get("tokenizer.json")?;
    let config_filename = repo.get("config.json")?;

    let filenames =
        candle_server_utils::hub_load_safetensors(&repo, "model.safetensors.index.json")?;
    println!("retrieved the files in {:?}", start.elapsed());

    // ------------ init model ------------
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();

    const IS_USE_CPU: bool = false;
    let device = candle_server_utils::device(IS_USE_CPU)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };


    let config: Config1 = serde_json::from_reader(std::fs::File::open(config_filename)?)?;
    let model = Model1::new(false, &config, vb)?;

    println!("loaded the model in {:?}", start.elapsed());

    // ------------ inference ------------
    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        TEMPERATURE,
        args.top_p,
        REPEAT_PENALTY,
        REPEAT_LAST_N,
        &device,
    );
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}
