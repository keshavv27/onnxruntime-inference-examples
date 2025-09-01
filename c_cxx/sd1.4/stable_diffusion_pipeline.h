#pragma once

#include <onnxruntime_cxx_api.h>
#include "lmscheduler.h"
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <filesystem>
#include <unordered_map>
#include <array>

namespace fs = std::filesystem;

class StableDiffusionPipeline {
public:
    struct GenerationConfig {
        std::string prompt;
        std::string negative_prompt = "";
        int num_inference_steps = 15;
        float guidance_scale = 7.5f;
        int height = 512;
        int width = 512;
        int64_t seed = -1;
        std::string output_path = "output.png";
    };

    explicit StableDiffusionPipeline(const std::string& model_path);
    ~StableDiffusionPipeline() = default;

    // Generate image from prompt
    bool generate(const GenerationConfig& config);

private:
    // CLIP Tokenizer constants
    static constexpr int MAX_SEQUENCE_LENGTH = 77;
    static constexpr int VOCAB_SIZE = 49408;
    static constexpr int PAD_TOKEN_ID = 49407;
    static constexpr int BOS_TOKEN_ID = 49406;
    static constexpr int EOS_TOKEN_ID = 49407;

    // Initialize models
    void initialize_models();

    // Tokenization and encoding
    std::vector<int32_t> tokenize(const std::string& text);
    std::vector<float> encode_prompt(const std::string& prompt, const std::string& negative_prompt = "");

    // Diffusion process
    std::vector<float> generate_initial_latents(int height, int width, int64_t seed, float init_noise_sigma);
    std::vector<float> denoise_latents(const std::vector<float>& text_embeddings,
                                     std::vector<float>& latents,
                                     int num_steps, float guidance_scale, LMSDiscreteScheduler& scheduler);

        // VAE encoding and decoding
    std::vector<float> encode_image(const std::vector<float>& image_data);
    std::vector<float> decode_latents(const std::vector<float>& latents);
    bool save_image(const std::vector<float>& image_data, const std::string& output_path,
                   int height, int width);

    // ONNX Runtime components
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "StableDiffusion"};
    Ort::SessionOptions text_session,unet_session,vae_encoder_session,vae_decoder_session,tokenizer_session;
    std::unique_ptr<Ort::Session> text_encoder;
    std::unique_ptr<Ort::Session> unet;
    std::unique_ptr<Ort::Session> vae_encoder;
    std::unique_ptr<Ort::Session> vae_decoder;
    std::unique_ptr<Ort::Session> tokenizer;
    // Model paths
    fs::path model_dir;

    // Random number generator
    std::mt19937 rng;

    // Constants and parameters
    const float vae_scale_factor = 0.18215f;
    const int latent_channels = 4;

};
