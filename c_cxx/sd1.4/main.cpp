#include "stable_diffusion_pipeline.h"
#include <iostream>
#include <string>
#include <chrono>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Required arguments:\n";
    std::cout << "  --model_path PATH     Path to the ONNX model directory\n";
    std::cout << "  --prompt \"TEXT\"       Text prompt for image generation\n\n";
    std::cout << "Optional arguments:\n";
    std::cout << "  --negative_prompt \"TEXT\"  Negative prompt (default: empty)\n";
    std::cout << "  --output PATH         Output image path (default: output.png)\n";
    std::cout << "  --steps N             Number of inference steps (default: 50)\n";
    std::cout << "  --guidance_scale F    Guidance scale (default: 7.5)\n";
    std::cout << "  --height N            Image height (default: 512)\n";
    std::cout << "  --width N             Image width (default: 512)\n";
    std::cout << "  --seed N              Random seed (default: random)\n";
    std::cout << "  --help                Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --model_path ./sd1.5 --prompt \"a beautiful sunset\"\n";
    std::cout << "  " << program_name << " --model_path ./sd1.5 --prompt \"a cat\" --steps 30 --seed 42\n\n";
}

std::string get_argument_value(int argc, char* argv[], const std::string& arg_name) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == arg_name) {
            return std::string(argv[i + 1]);
        }
    }
    return "";
}

bool has_argument(int argc, char* argv[], const std::string& arg_name) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == arg_name) {
            return true;
        }
    }
    return false;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Stable Diffusion 1.5 ONNX Runtime Pipeline ===\n";
    std::cout << "Using NvTensorRtRtx Execution Provider\n\n";

    // Check for help
    if (argc < 2 || has_argument(argc, argv, "--help") || has_argument(argc, argv, "-h")) {
        print_usage(argv[0]);
        return 0;
    }

    // Parse command line arguments
    std::string model_path = get_argument_value(argc, argv, "--model_path");
    std::string prompt = get_argument_value(argc, argv, "--prompt");

    if (model_path.empty()) {
        std::cerr << "Error: --model_path is required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    if (prompt.empty()) {
        std::cerr << "Error: --prompt is required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // Parse optional arguments
    StableDiffusionPipeline::GenerationConfig config;
    config.prompt = prompt;
    config.negative_prompt = get_argument_value(argc, argv, "--negative_prompt");
    config.output_path = get_argument_value(argc, argv, "--output");
    if (config.output_path.empty()) {
        config.output_path = "output.png";
    }

    std::string steps_str = get_argument_value(argc, argv, "--steps");
    if (!steps_str.empty()) {
        try {
            config.num_inference_steps = std::stoi(steps_str);
            if (config.num_inference_steps <= 0 || config.num_inference_steps > 200) {
                std::cerr << "Warning: steps should be between 1 and 200, using default (50)\n";
                config.num_inference_steps = 50;
            }
        } catch (const std::exception&) {
            std::cerr << "Warning: invalid steps value, using default (50)\n";
            config.num_inference_steps = 50;
        }
    }

    std::string guidance_str = get_argument_value(argc, argv, "--guidance_scale");
    if (!guidance_str.empty()) {
        try {
            config.guidance_scale = std::stof(guidance_str);
            if (config.guidance_scale < 1.0f || config.guidance_scale > 20.0f) {
                std::cerr << "Warning: guidance_scale should be between 1.0 and 20.0, using default (7.5)\n";
                config.guidance_scale = 7.5f;
            }
        } catch (const std::exception&) {
            std::cerr << "Warning: invalid guidance_scale value, using default (7.5)\n";
            config.guidance_scale = 7.5f;
        }
    }

    std::string height_str = get_argument_value(argc, argv, "--height");
    if (!height_str.empty()) {
        try {
            config.height = std::stoi(height_str);
            if (config.height % 64 != 0 || config.height < 64 || config.height > 1024) {
                std::cerr << "Warning: height should be divisible by 64 and between 64 and 1024, using default (512)\n";
                config.height = 512;
            }
        } catch (const std::exception&) {
            std::cerr << "Warning: invalid height value, using default (512)\n";
            config.height = 512;
        }
    }

    std::string width_str = get_argument_value(argc, argv, "--width");
    if (!width_str.empty()) {
        try {
            config.width = std::stoi(width_str);
            if (config.width % 64 != 0 || config.width < 64 || config.width > 1024) {
                std::cerr << "Warning: width should be divisible by 64 and between 64 and 1024, using default (512)\n";
                config.width = 512;
            }
        } catch (const std::exception&) {
            std::cerr << "Warning: invalid width value, using default (512)\n";
            config.width = 512;
        }
    }

    std::string seed_str = get_argument_value(argc, argv, "--seed");
    if (!seed_str.empty()) {
        try {
            config.seed = std::stoll(seed_str);
        } catch (const std::exception&) {
            std::cerr << "Warning: invalid seed value, using random seed\n";
            config.seed = -1;
        }
    }

    try {
        // Initialize pipeline
        std::cout << "Model path: " << model_path << "\n";
        StableDiffusionPipeline pipeline(model_path);

        // Generate image
        bool success = pipeline.generate(config);

        if (success) {
            std::cout << "\n Generation completed successfully!\n";
            return 0;
        } else {
            std::cerr << "\n Generation failed!\n";
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "\n Fatal error: " << e.what() << "\n";
        std::cerr << "\nTroubleshooting:\n";
        std::cerr << "1. Check that the model path exists and contains the required ONNX files\n";
        std::cerr << "2. Ensure you have a compatible GPU and NVIDIA drivers installed\n";
        std::cerr << "3. Verify that the NvTensorRtRtx execution provider is available\n";
        std::cerr << "4. Check that you have enough GPU memory for the model\n";
        return 1;
    }
}

