#include "stable_diffusion_pipeline.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <regex>

#define STB_IMAGE_WRITE_FILENAME_STRING
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

StableDiffusionPipeline::StableDiffusionPipeline(const std::string& model_path)
    : model_dir(model_path) {

    std::cout << "Initializing Stable Diffusion Pipeline...\n";

    // Check ONNX Runtime version compatibility
    auto ort_version = Ort::GetVersionString();
    std::cout << "ONNX Runtime version: " << ort_version << "\n";

    // Check available API versions
    try {
        auto api_version = OrtGetApiBase()->GetVersionString();
        std::cout << "ONNX Runtime API version string: " << api_version << "\n";
    } catch (const std::exception& e) {
        std::cout << "Could not get API version: " << e.what() << "\n";
    }

    // Display available execution providers
    auto available_providers = Ort::GetAvailableProviders();
    std::cout << "Available execution providers: ";
    for (const auto& provider : available_providers) {
        std::cout << provider << " ";
    }
    std::cout << "\n";

    // Initialize random number generator
    rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());

    //initialize models
    initialize_models();

    std::cout << "Pipeline initialized successfully!\n";
}

void StableDiffusionPipeline::initialize_models() {
    std::cout << "Loading ONNX models...\n";
    std::unordered_map<std::string, std::string> provider_options;
    provider_options["device_id"] = "0";
    const OrtApi* g_ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    try {
        //Load tokenizer
        auto tokenizer_path = model_dir / "tokenizer" / "cliptokenizer.onnx";
        auto extension_dll = model_dir / "ortextensions.dll";
        void* handle;
        try {
          Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary(tokenizer_session, extension_dll.string().c_str(), &handle));
        } catch (Ort::Exception& e) {
          std::cerr << "Error during customops library loading: " << e.what() << std::endl;
        }
        if (!fs::exists(tokenizer_path)) {
          throw std::runtime_error("Clip tokenizer model not found: " + tokenizer_path.string());
        }
        
        tokenizer = std::make_unique<Ort::Session>(env, tokenizer_path.wstring().c_str(), tokenizer_session);
        std::cout << "  Tokenizer loaded\n";

        // Load text encoder
        auto text_encoder_path = model_dir / "text_encoder" / "model.onnx";
        if (!fs::exists(text_encoder_path)) {
            throw std::runtime_error("Text encoder model not found: " + text_encoder_path.string());
        }
        g_ort_api->AddFreeDimensionOverrideByName(text_session, "batch", 1);
        g_ort_api->AddFreeDimensionOverrideByName(text_session, "channels", 3);
        g_ort_api->AddFreeDimensionOverrideByName(text_session, "height", 512);
        g_ort_api->AddFreeDimensionOverrideByName(text_session, "width", 512);
        text_session.AppendExecutionProvider("NvTensorRTRTXExecutionProvider", provider_options);
        text_encoder = std::make_unique<Ort::Session>(env,
            text_encoder_path.wstring().c_str(),
            text_session);
        std::cout << "Text encoder loaded\n";

        // Load UNet
        auto unet_path = model_dir / "unet" / "model.onnx";
        if (!fs::exists(unet_path)) {
            throw std::runtime_error("UNet model not found: " + unet_path.string());
        }
        g_ort_api->AddFreeDimensionOverrideByName(unet_session, "batch", 2);
        g_ort_api->AddFreeDimensionOverrideByName(unet_session, "channels", 4);
        g_ort_api->AddFreeDimensionOverrideByName(unet_session, "height", 512 / 8);
        g_ort_api->AddFreeDimensionOverrideByName(unet_session, "width", 512 / 8);
        g_ort_api->AddFreeDimensionOverrideByName(unet_session, "sequence", 77);
        unet_session.AppendExecutionProvider("NvTensorRTRTXExecutionProvider", provider_options);
        unet = std::make_unique<Ort::Session>(env,
            unet_path.wstring().c_str(),
            unet_session);
        std::cout << "UNet loaded\n";

        // Load VAE decoder
        auto vae_decoder_path = model_dir / "vae_decoder" / "model.onnx";
        if (!fs::exists(vae_decoder_path)) {
            throw std::runtime_error("VAE decoder model not found: " + vae_decoder_path.string());
        }
        g_ort_api->AddFreeDimensionOverrideByName(vae_decoder_session, "batch", 1);
        g_ort_api->AddFreeDimensionOverrideByName(vae_decoder_session, "channels", 4);
        g_ort_api->AddFreeDimensionOverrideByName(vae_decoder_session, "height", 512 / 8);
        g_ort_api->AddFreeDimensionOverrideByName(vae_decoder_session, "width", 512 / 8);
        vae_decoder_session.AppendExecutionProvider("NvTensorRTRTXExecutionProvider", provider_options);
        vae_decoder = std::make_unique<Ort::Session>(env,
            vae_decoder_path.wstring().c_str(),
            vae_decoder_session);
        std::cout << "VAE decoder loaded\n";

    } catch (const Ort::Exception& e) {
        throw std::runtime_error("Failed to load ONNX models: " + std::string(e.what()));
    }
}

std::vector<int32_t> StableDiffusionPipeline::tokenize(const std::string& text) {

    //auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    auto allocator = Ort::AllocatorWithDefaultOptions();
    std::vector<int64_t> string_shape = {1};
    std::vector<const char*> text_vec;
    text_vec.push_back(text.c_str());

    Ort::Value string_input_tensor =
        Ort::Value::CreateTensor(allocator,/* text_vec.data(), sizeof(std::string),*/
                                 string_shape.data(), string_shape.size(),ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    auto status = Ort::GetApi().FillStringTensor(static_cast<OrtValue*>(string_input_tensor), text_vec.data(), 1U);

    std::vector<const char*> input_names{"string_input"};
    std::vector<const char*> output_names{"input_ids", "attention_mask"};
    
    auto outputs = tokenizer->Run(Ort::RunOptions{nullptr}, input_names.data(), &string_input_tensor,
                                  tokenizer->GetInputCount(), output_names.data(), tokenizer->GetOutputCount());

    
    int64_t* tokens_raw = outputs[0].GetTensorMutableData<int64_t>();
    auto tokens_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t tokens_size = 1;
    for (auto dim : tokens_shape) {
      tokens_size *= dim;
    }
    std::vector<int64_t> tokens(tokens_raw, tokens_raw + tokens_size);
 
    // Cast to int32
    std::vector<int32_t> input_ids;
    for (auto id : tokens) {
      input_ids.push_back(static_cast<int32_t>(id));
    }

    const int model_max_length = 77;
    const int64_t pad_token_id = 49407;

    // Check if input is too long
    if (input_ids.size() > model_max_length) {
      throw std::invalid_argument(
          "Input text is too long. Maximum allowed tokens: " + std::to_string(model_max_length) +
          ", but received: " + std::to_string(input_ids.size()) + ".");
    }

    // Pad array with 49407 until length is model_max_length
    if (input_ids.size() < model_max_length) {
      size_t padding_needed = model_max_length - input_ids.size();
      input_ids.resize(model_max_length, pad_token_id);
    }

    return input_ids;
}

std::vector<float> StableDiffusionPipeline::encode_prompt(const std::string& prompt,
                                                         const std::string& negative_prompt) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                   OrtMemType::OrtMemTypeDefault);

    // Tokenize positive prompt
    auto input_ids = tokenize(prompt);
    
    // Create input tensor
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(MAX_SEQUENCE_LENGTH)};
    Ort::Value input_tensor = Ort::Value::CreateTensor<int32_t>(
        memory_info, input_ids.data(), input_ids.size(),
        input_shape.data(), input_shape.size());

    // Run text encoder
    const char* input_names[] = {"input_ids"};
    const char* output_names[] = {"last_hidden_state"};

    auto outputs = text_encoder->Run(Ort::RunOptions{nullptr},
                                   input_names, &input_tensor, 1,
                                   output_names, 1);

    // Get text embeddings
    float* output_data = outputs[0].GetTensorMutableData<float>();
    auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t embedding_size = 1;
    for (auto dim : output_shape) {
        embedding_size *= dim;
    }

    std::vector<float> text_embeddings(output_data, output_data + embedding_size);

    // Handle negative prompt
    std::vector<float> uncond_embeddings;
    if (!negative_prompt.empty()) {
        auto neg_tokens = tokenize(negative_prompt);
        std::vector<int64_t> neg_input_ids(neg_tokens.begin(), neg_tokens.end());

        Ort::Value neg_input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, neg_input_ids.data(), neg_input_ids.size(),
            input_shape.data(), input_shape.size());

        auto neg_outputs = text_encoder->Run(Ort::RunOptions{nullptr},
                                           input_names, &neg_input_tensor, 1,
                                           output_names, 1);

        float* neg_output_data = neg_outputs[0].GetTensorMutableData<float>();
        uncond_embeddings.assign(neg_output_data, neg_output_data + embedding_size);
    } else {
        // Use empty string for unconditional embeddings
        //auto empty_tokens = tokenize("");
        //std::vector<int64_t> empty_input_ids_long(empty_tokens.begin(), empty_tokens.end());
        std::vector<int32_t> empty_input_ids{49406};
        size_t maxlength = 77;
        for (int i = 0; i < maxlength - 1; i++) empty_input_ids.push_back(49407);

        Ort::Value empty_input_tensor = Ort::Value::CreateTensor<int32_t>(
            memory_info, empty_input_ids.data(), empty_input_ids.size(),
            input_shape.data(), input_shape.size());

        auto empty_outputs = text_encoder->Run(Ort::RunOptions{nullptr},
                                             input_names, &empty_input_tensor, 1,
                                             output_names, 1);

        float* empty_output_data = empty_outputs[0].GetTensorMutableData<float>();
        uncond_embeddings.assign(empty_output_data, empty_output_data + embedding_size);
    }

    // Concatenate unconditional and conditional embeddings
    std::vector<float> combined_embeddings;
    combined_embeddings.reserve(uncond_embeddings.size() + text_embeddings.size());
    combined_embeddings.insert(combined_embeddings.end(), uncond_embeddings.begin(), uncond_embeddings.end());
    combined_embeddings.insert(combined_embeddings.end(), text_embeddings.begin(), text_embeddings.end());

    return combined_embeddings;
}

std::vector<float> StableDiffusionPipeline::generate_initial_latents(int height, int width, int64_t seed, float init_noise_sigma) {
    if (seed >= 0) {
        rng.seed(seed);
    }

    const int latent_height = height / 8;
    const int latent_width = width / 8;
    const size_t total_size = 1 * latent_channels * latent_height * latent_width;

    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> latents(total_size);

    for (auto& val : latents) {
        val = dist(rng) * init_noise_sigma;
    }

    return latents;
}

std::vector<float> StableDiffusionPipeline::denoise_latents(
    const std::vector<float>& text_embeddings,
    std::vector<float>& latents,
    int num_steps, float guidance_scale, LMSDiscreteScheduler& scheduler) {

   // set_timesteps(num_steps);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                   OrtMemType::OrtMemTypeDefault);

    const int latent_height = 64; // 512 / 8
    const int latent_width = 64;  // 512 / 8
    const int embedding_dim = 768; // CLIP embedding dimension


    for (size_t i = 0; i < scheduler.get_timesteps().size(); ++i) {
      int t = (scheduler.get_timesteps())[i];

        // Duplicate latents for classifier-free guidance
        std::vector<float> latent_model_input;
        latent_model_input.reserve(latents.size() * 2);
        latent_model_input.insert(latent_model_input.end(), latents.begin(), latents.end());
        latent_model_input.insert(latent_model_input.end(), latents.begin(), latents.end());

        // Create input tensors
        std::vector<int64_t> latent_shape = {2, latent_channels, latent_height, latent_width};
        std::vector<int64_t> timestep_shape = {2};
        std::vector<int64_t> embedding_shape = {2, MAX_SEQUENCE_LENGTH, embedding_dim};

        latent_model_input = scheduler.scale_input(latent_model_input, t);

        Ort::Value latent_tensor = Ort::Value::CreateTensor<float>(
            memory_info, latent_model_input.data(), latent_model_input.size(),
            latent_shape.data(), latent_shape.size());

        std::vector<int64_t> timestep_data = {static_cast<int64_t>(t), static_cast<int64_t>(t)};
        Ort::Value timestep_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, timestep_data.data(), timestep_data.size(),
            timestep_shape.data(), timestep_shape.size());

        std::vector<float> embedding_data(text_embeddings);
        Ort::Value embedding_tensor = Ort::Value::CreateTensor<float>(
            memory_info, embedding_data.data(), embedding_data.size(),
            embedding_shape.data(), embedding_shape.size());

        // Run UNet
        const char* input_names[] = {"sample", "timestep", "encoder_hidden_states"};
        const char* output_names[] = {"out_sample"};

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(latent_tensor));
        input_tensors.push_back(std::move(timestep_tensor));
        input_tensors.push_back(std::move(embedding_tensor));


        auto outputs = unet->Run(Ort::RunOptions{nullptr},
                               input_names, input_tensors.data(), 3,
                               output_names, 1);

        // Get noise prediction
        float* noise_pred_data = outputs[0].GetTensorMutableData<float>();
        auto noise_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

        size_t single_latent_size = latents.size();

        // Split noise prediction for classifier-free guidance
        std::vector<float> noise_pred_uncond(noise_pred_data,
                                           noise_pred_data + single_latent_size);
        std::vector<float> noise_pred_text(noise_pred_data + single_latent_size,
                                         noise_pred_data + 2 * single_latent_size);

        // Apply classifier-free guidance
        std::vector<float> noise_pred(single_latent_size);
        for (size_t j = 0; j < single_latent_size; ++j) {
            noise_pred[j] = noise_pred_uncond[j] + guidance_scale *
                           (noise_pred_text[j] - noise_pred_uncond[j]);
        }

        // LMS scheduler step
        latents = scheduler.step(noise_pred, i, latents);

        if ((i + 1) % 10 == 0) {
          std::cout << "Denoising step " << (i + 1) << "/" << scheduler.get_timesteps().size() << std::endl;
        }
    }

    return latents;
}

std::vector<float> StableDiffusionPipeline::encode_image(const std::vector<float>& image_data) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                   OrtMemType::OrtMemTypeDefault);

    // Assuming image_data is in HWC format and normalized to [0, 1]
    // Convert to CHW format and normalize to [-1, 1]
    int height = 512; 
    int width = 512; 
    int channels = 3;

    std::vector<float> chw_image(channels * height * width);

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int hwc_idx = h * width * channels + w * channels + c;
                int chw_idx = c * height * width + h * width + w;
                // Normalize from [0, 1] to [-1, 1]
                chw_image[chw_idx] = 2.0f * image_data[hwc_idx] - 1.0f;
            }
        }
    }

    // Create input tensor
    std::vector<int64_t> input_shape = {1, channels, height, width};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, chw_image.data(), chw_image.size(),
        input_shape.data(), input_shape.size());

    // Run VAE encoder
    const char* input_names[] = {"sample"};
    const char* output_names[] = {"latent_sample"};

    auto outputs = vae_encoder->Run(Ort::RunOptions{nullptr},
                                  input_names, &input_tensor, 1,
                                  output_names, 1);

    // Get encoded latents
    float* latent_data = outputs[0].GetTensorMutableData<float>();
    auto latent_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t latent_size = 1;
    for (auto dim : latent_shape) {
        latent_size *= dim;
    }

    std::vector<float> latents(latent_data, latent_data + latent_size);

    // Scale latents
    for (auto& latent : latents) {
        latent *= vae_scale_factor;
    }

    return latents;
}

std::vector<float> StableDiffusionPipeline::decode_latents(const std::vector<float>& latents) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                   OrtMemType::OrtMemTypeDefault);

    // Scale latents
    std::vector<float> scaled_latents(latents.size());
    for (size_t i = 0; i < latents.size(); ++i) {
        scaled_latents[i] = latents[i] / vae_scale_factor;
    }

    // Create input tensor
    std::vector<int64_t> latent_shape = {1, latent_channels, 64, 64}; // 512/8 = 64
    Ort::Value latent_tensor = Ort::Value::CreateTensor<float>(
        memory_info, scaled_latents.data(), scaled_latents.size(),
        latent_shape.data(), latent_shape.size());

    // Run VAE decoder
    const char* input_names[] = {"latent_sample"};
    const char* output_names[] = {"sample"};

    auto outputs = vae_decoder->Run(Ort::RunOptions{nullptr},
                                  input_names, &latent_tensor, 1,
                                  output_names, 1);

    // Get decoded image
    float* image_data = outputs[0].GetTensorMutableData<float>();
    auto image_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t image_size = 1;
    for (auto dim : image_shape) {
        image_size *= dim;
    }

    std::vector<float> image(image_data, image_data + image_size);

    // Normalize image data from [-1, 1] to [0, 1]
    for (auto& pixel : image) {
        pixel = (pixel + 1.0f) / 2.0f;
        pixel = std::clamp(pixel, 0.0f, 1.0f);
    }

    return image;
}

bool StableDiffusionPipeline::save_image(const std::vector<float>& image_data,
                                       const std::string& output_path,
                                       int height, int width) {
    // Convert float data to uint8
    std::vector<uint8_t> image_bytes(image_data.size());
    for (size_t i = 0; i < image_data.size(); ++i) {
        image_bytes[i] = static_cast<uint8_t>(image_data[i] * 255.0f);
    }

    // The image data is in CHW format (Channel, Height, Width)
    // We need to convert to HWC format (Height, Width, Channel)
    std::vector<uint8_t> hwc_image(height * width * 3);

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < 3; ++c) {
                int chw_idx = c * height * width + h * width + w;
                int hwc_idx = h * width * 3 + w * 3 + c;
                hwc_image[hwc_idx] = image_bytes[chw_idx];
            }
        }
    }

    // Save using STB
    int result = stbi_write_png(output_path.c_str(), width, height, 3,
                               hwc_image.data(), width * 3);

    return result != 0;
}

bool StableDiffusionPipeline::generate(const GenerationConfig& config) {
    std::cout << "\n=== Starting Image Generation ===\n";
    std::cout << "Prompt: \"" << config.prompt << "\"\n";
    std::cout << "Steps: " << config.num_inference_steps << ", Guidance: " << config.guidance_scale << "\n";
    std::cout << "Resolution: " << config.width << "x" << config.height << "\n";
    if (config.seed >= 0) {
        std::cout << "Seed: " << config.seed << "\n";
    }
    std::cout << "=================================\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    LMSDiscreteScheduler scheduler(config.num_inference_steps);

    try {
        // 1. Encode prompt
        std::cout << "Encoding prompt...\n";
        auto text_embeddings = encode_prompt(config.prompt, config.negative_prompt);

        // 2. Generate initial latents
        std::cout << "Generating initial latents...\n";
        auto latents = generate_initial_latents(config.height, config.width, config.seed,scheduler.get_init_noise_sigma());

        // 3. Denoise latents
        std::cout << "Starting denoising process...\n";
        latents = denoise_latents(text_embeddings, latents,
                                config.num_inference_steps, config.guidance_scale, scheduler);

        // 4. Decode latents to image
        std::cout << "Decoding latents to image...\n";
        auto image = decode_latents(latents);

        // 5. Save image
        std::cout << "Saving image...\n";
        bool success = save_image(image, config.output_path, config.height, config.width);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();

        if (success) {
            std::cout << "\nImage generated successfully!\n";
            std::cout << "Saved to: " << config.output_path << "\n";
            std::cout << "Generation time: " << duration / 1000.0 << " seconds\n";
        } else {
          std::cerr << "Failed to save image\n ";
            return false;
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error during generation: " << e.what() << "\n";
        return false;
    }
}
