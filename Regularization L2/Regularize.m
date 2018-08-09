function result = Regularize(Weights_Array, lambda)

   % L2 Regularization
   result = Linear_Combination(Weights_Array, - lambda );
