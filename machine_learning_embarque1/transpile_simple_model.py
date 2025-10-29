import sys
import os
import joblib
import numpy as np


def generate_logistic_regression(thetas):
    n_features = len(thetas) - 1

    code = "#include <stdio.h>\n\n"
    code += """
float exp_approx(float x, int n_term)
{
    float sum = 0.0;
    
    float x_power = 1.0;
    int i_facto = 1;
    
    for (int i = 1; i < n_term + 2; i++)
    {
        sum += x_power / i_facto;
        x_power *= x;
        i_facto *= i;
    }
    return sum;
}

float linear_regression_prediction(float *features, float *thetas, int n_parameters)
{
    float prediction = thetas[0];
    for (int i = 0; i < n_parameters; i++) {
        prediction += features[i] * thetas[i + 1];
    }
    return prediction;
}

float sigmoid(float x)
{
    return 1 / (1 + exp_approx(-x, 10));
}

"""
    code += "float logistic_regression(float *features, int n_feature)\n"
    code += "{\n"
    code += f"    float thetas[{len(thetas)}] = {{{', '.join([f'{t:.8f}f' for t in thetas])}}};\n"
    code += "    return sigmoid(linear_regression_prediction(features, thetas, n_feature));\n"
    code += "}\n\n"

    features_init = ", ".join(["1.0f"] * n_features)

    code += f"""int main() 
{{
    float features[{n_features}] = {{{features_init}}};
    float y = logistic_regression(features, {n_features});
    printf("Prediction: %f\\n", y);
    return 0;
}}
"""
    return code


def decision_tree(thetas):
    n_features = len(thetas) - 1

    code = "#include <stdio.h>\n\n"
    code += "int decision_tree(float *features, int n_features)\n"
    code += "{\n"
    code += "   int res = 0;\n"
    code += "   for (int i = 0; i < n_features; i++) {\n"
    code += "       res |= (features[i] > 0);\n"
    code += "   }\n"
    code += "   return !res;\n"
    code += "}\n\n"

    features_init = ", ".join(["1.0f"] * n_features)

    code += f"""int main() 
{{
    float features[{n_features}] = {{{features_init}}};
    int y = decision_tree(features, {n_features});
    printf("Prediction: %d\\n", y);
    return 0;
}}
"""
    return code


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage : python transpile_simple_model.py model.joblib output.c")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]

    model = joblib.load(model_path)
    coefs = model.coef_.ravel()
    intercept = model.intercept_

    thetas = [intercept] + coefs.tolist()

    c_code = decision_tree(thetas)

    with open(output_path, "w") as f:
        f.write(c_code)
    
    compile_cmd = f"gcc -std=c99 -Werror -Wvla -Wextra -pedantic -Wall {output_path} -o {output_path.split('.', 1)[0]}"
    os.system(compile_cmd)
