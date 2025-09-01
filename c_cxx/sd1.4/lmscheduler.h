#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <iostream>

class LMSDiscreteScheduler {
    int num_train_timesteps;
    std::vector<float> alphas_cumprod;
    std::vector<float> sigmas;
    std::vector<std::vector<float>> derivatives;
    std::string prediction_type;
    std::vector<int> timesteps;
    float init_noise_sigma;

    // Math helpers for vectors
    static std::vector<float> add_vec(const std::vector<float>& a, const std::vector<float>& b) {
        assert(a.size() == b.size());
        std::vector<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] + b[i];
        return result;
    }
    static std::vector<float> sub_vec(const std::vector<float>& a, const std::vector<float>& b) {
        assert(a.size() == b.size());
        std::vector<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] - b[i];
        return result;
    }
    static std::vector<float> mul_vec_scalar(const std::vector<float>& a, float s) {
        std::vector<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] * s;
        return result;
    }
    static std::vector<float> div_vec_scalar(const std::vector<float>& a, float s) {
        std::vector<float> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] / s;
        return result;
    }
    static std::vector<double> linspace(double start, double end, int num) {
        std::vector<double> result(num);
        if (num == 1) {
            result[0] = start;
            return result;
        }
        double step = (end - start) / (num - 1);
        for (int i = 0; i < num; ++i) result[i] = start + step * i;
        return result;
    }
    static std::vector<double> interpolate(const std::vector<double>& tsteps, const std::vector<float>& sigmas) {
        std::vector<double> result(tsteps.size() + 1);
        std::vector<double> range(sigmas.size());
        for (size_t i = 0; i < sigmas.size(); ++i) range[i] = static_cast<double>(i);
        for (size_t i = 0; i < tsteps.size(); ++i) {
            auto it = std::lower_bound(range.begin(), range.end(), tsteps[i]);
            int index = int(it - range.begin());
            if (index < sigmas.size() && range[index] == tsteps[i]) {
                result[i] = sigmas[index];
            } else if (index == 0) {
                result[i] = sigmas[0];
            } else if (index == sigmas.size()) {
                result[i] = sigmas.back();
            } else {
                double t = (tsteps[i] - range[index - 1]) / (range[index] - range[index - 1]);
                result[i] = sigmas[index - 1] + t * (sigmas[index] - sigmas[index - 1]);
            }
        }
        return result;
    }
    // Simple fixed-step Riemann integration for demonstration (100 intervals)
    double integrate(const std::function<double(double)>& f, double a, double b) {
        int n = 100;
        double h = (b - a) / n;
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += f(a + i * h) * h;
        return sum;
    }
    double get_lms_coefficient(int order, int t, int current_order) {
        auto lms_derivative = [&](double tau) {
            double prod = 1.0;
            for (int k = 0; k < order; ++k) {
                if (current_order == k) continue;
                prod *= (tau - sigmas[t - k]) / (sigmas[t - current_order] - sigmas[t - k]);
            }
            return prod;
        };
        return integrate(lms_derivative, sigmas[t], sigmas[t + 1]);
    }
public:
    LMSDiscreteScheduler(int num_inference_steps, int num_train_timesteps = 1000,
                        float beta_start = 0.00085f, float beta_end = 0.012f,
                        const std::string& beta_schedule = "scaled_linear",
                        const std::string& prediction_type = "epsilon",
                        const std::vector<float>* trained_betas = nullptr)
        : num_train_timesteps(num_train_timesteps), prediction_type(prediction_type)
    {
        std::vector<float> betas;
        // Beta schedule logic
        if (trained_betas) {
            betas = *trained_betas;
        } else if (beta_schedule == "linear") {
            betas.resize(num_train_timesteps);
            for (int i = 0; i < num_train_timesteps; ++i)
                betas[i] = beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1);
        } else if (beta_schedule == "scaled_linear") {
            betas.resize(num_train_timesteps);
            double start = std::sqrt(beta_start), end = std::sqrt(beta_end);
            auto lin = linspace(start, end, num_train_timesteps);
            for (int i = 0; i < num_train_timesteps; ++i)
                betas[i] = lin[i] * lin[i];
        } else {
            betas = std::vector<float>(num_train_timesteps, beta_start);
        }
        std::vector<float> alphas(num_train_timesteps);
        for (int i = 0; i < num_train_timesteps; ++i)
            alphas[i] = 1.0f - betas[i];
        alphas_cumprod.resize(num_train_timesteps);
        float prod = 1.0f;
        for (int i = 0; i < num_train_timesteps; ++i) {
            prod *= alphas[i];
            alphas_cumprod[i] = prod;
        }
        std::vector<float> tmp_sigmas(num_train_timesteps);
        for (int i = 0; i < num_train_timesteps; ++i)
            tmp_sigmas[i] = std::sqrt((1.f - alphas_cumprod[i]) / alphas_cumprod[i]);
        std::reverse(tmp_sigmas.begin(), tmp_sigmas.end());
        init_noise_sigma = *std::max_element(tmp_sigmas.begin(), tmp_sigmas.end());
        // Time steps
        auto tsteps = linspace(0, num_train_timesteps - 1, num_inference_steps);
        timesteps.assign(tsteps.rbegin(), tsteps.rend());
        auto interpolated = interpolate(tsteps, tmp_sigmas);
        sigmas.assign(interpolated.begin(), interpolated.end());
    }

    // Pre-scaling input; not commonly used in LMS/SD C++ pipelines but provided for completeness
    std::vector<float> scale_input(const std::vector<float>& sample, int timestep) {
        auto it = std::find(timesteps.begin(), timesteps.end(), timestep);
        int step_index = int(it - timesteps.begin());
        float sigma = sigmas[step_index];
        sigma = std::sqrt(sigma * sigma + 1.0f);
        return div_vec_scalar(sample, sigma);
    }

    std::vector<float> step(const std::vector<float>& model_output,
                            int step_index,
                            const std::vector<float>& sample,
                            int order = 4) {
        float sigma = sigmas[step_index];
        // Compute pred_original_sample per element
        std::vector<float> pred_original_sample(sample.size());
        if (prediction_type == "epsilon") {
            for (size_t i = 0; i < sample.size(); ++i)
                pred_original_sample[i] = sample[i] - sigma * model_output[i];
        } else {
            std::cerr << "Warning: Unsupported prediction_type " << prediction_type << std::endl;
            return std::vector<float>(sample.size(), 0.0f);
        }
        // Derivative: (sample - pred_original_sample) / sigma
        std::vector<float> derivative(sample.size());
        for (size_t i = 0; i < sample.size(); ++i)
            derivative[i] = (sample[i] - pred_original_sample[i]) / sigma;
        derivatives.push_back(derivative);
        if (derivatives.size() > size_t(order))
            derivatives.erase(derivatives.begin());
        int actual_order = std::min(step_index + 1, order);
        std::vector<double> lms_coeffs(actual_order);
        for (int curr_order = 0; curr_order < actual_order; ++curr_order)
            lms_coeffs[curr_order] = get_lms_coefficient(actual_order, step_index, curr_order);
        // Dot product (reversed derivatives to match Python/C# logic)
        std::vector<float> lms_result(sample.size(), 0.0f);
        for (int m = 0; m < actual_order; ++m) {
            const auto& der = derivatives[derivatives.size() - 1 - m];
            auto scaled = mul_vec_scalar(der, static_cast<float>(lms_coeffs[m]));
            for (size_t i = 0; i < sample.size(); ++i) lms_result[i] += scaled[i];
        }
        return add_vec(sample, lms_result);
    }
    const std::vector<int>& get_timesteps() const { return timesteps; }
    float get_init_noise_sigma() const { return init_noise_sigma; }
    float get_sigma(int step_index) const { return sigmas[step_index]; }
};
