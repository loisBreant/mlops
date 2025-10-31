#include <stdio.h>

int decision_tree(float *features, int n_features)
{
   int res = 0;
   for (int i = 0; i < n_features; i++) {
       res |= (features[i] > 0);
   }
   return !res;
}

int main() 
{
    float features[3] = {0, 0, 0};
    int y = decision_tree(features, 3);
    printf("Prediction: %d\n", y);
    return 0;
}
