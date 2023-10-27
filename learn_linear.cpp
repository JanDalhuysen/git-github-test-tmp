#include <iostream>
#include "learn_linear_to_c_1.h"

// using namespace std;

int main()
{
    float input[1] = {10};

    Eloquent::ML::Port::LinearRegression model;

    int a = model.predict(input);
    float b = model.predict(input);
    double c = model.predict(input);
    
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
}
