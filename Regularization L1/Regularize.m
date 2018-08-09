function result = Regularize(Weights_Array, lambda)

    % L1 Regularization
    Regularizer = ApplyFunction(@sign, Weights_Array);
    result = Linear_Combination(Regularizer, -lambda);
    