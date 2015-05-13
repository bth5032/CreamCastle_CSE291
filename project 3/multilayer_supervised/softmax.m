function [p]  = softmax(z)
    numerator = exp(z);
    denominator = sum(numerator);
    p = bsxfun(@rdivide, numerator, denominator);
end