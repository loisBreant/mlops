float linear_regression_prediction(float *features, float *thetas, int n_parameters)
{
    float prediction = thetas[0];
    for (int i = 0; i < n_parameters; i++) {
        prediction += features[i] * thetas[i + 1];
    }
    return prediction;
}

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

float sigmoid(float x)
{
    return 1 / (1 + exp_approx(-x, 10));
}

float logistic_regression(float* features, float* thetas, int n_parameter)
{
    return sigmoid(linear_regression_prediction(features, thetas, n_parameter));
}