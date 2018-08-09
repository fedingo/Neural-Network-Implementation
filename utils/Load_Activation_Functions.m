
%% Activation functions

Logistic = @(x) 1./(1+exp(-x));
Logistic_derivative = @(x) Logistic(x).*(1-Logistic(x));

TanH = @(x) 2 ./ ( 1 + exp(-2 * x) ) -1;
TanH_derivative = @(x) 1 - TanH(x) .^ 2;

SoftPlus = @(x) log( 1+exp(x) );
SoftPlus_derivative = @(x) 1 ./ ( 1 + exp(-x) );

Relu = @(x) max(0,x);
Relu_derivative = @(x) max(0, sign(x));


%% The chosen Activation Function with its derivative

Activation = TanH;
Diff_Activation = TanH_derivative;

%Activation = Relu;
%Diff_Activation = Relu_derivative;