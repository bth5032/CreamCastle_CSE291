function [p]  = softmax(z)
    numerator = exp(z);
    denominator = sum(numerator);
    p = numerator/denominator;
end