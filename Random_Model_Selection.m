function Result = Random_Model_Selection (Input, Output, K, Ranges, Limit, regression)
    
    % K = number of folds for Cross-validation
    % Ranges is an array of 4x2
    % Limit = number of random tries
    % Regression = boolean variable
    
    EarlyStoppingError = 0.01;
    if nargin < 5
        Limit = 100;
    end
    if nargin < 6
        regression = false;
    end
    
    % Number of Input samples and Output Samples should be equal
    if size(Input,1) ~= size(Output,1)
        fprintf("Input and Output are not of the same size!\n");
        return;
    end
    
    NumberOfSamples = size(Input,1);
    FoldSize = floor(NumberOfSamples/K);
    shuffle = randperm(NumberOfSamples);

    BestGuy = {};       % best configuration of parameters up to now
    BestError = 1000;   % best error found up to now
    Errors = zeros(Limit);
    tic;

    for RandomSearch = 1:Limit
        
        randomGuy = GenerateParameters(Ranges); % generate random parameters
      
        LCurves = zeros(K,1);
        VCurves = zeros(K,1);
        
        % Use the function Cross-Validation
        for index = 0:K-1
            % Training fold = difference between the all dataset shuffled and the TestFold
            TestFold = shuffle(FoldSize*index+1:FoldSize*(index+1));
            TrainingFold = setdiff(shuffle, TestFold);

            [LC, VC] = Training(  Input(TrainingFold,:), ...
                                        Output(TrainingFold,:), ...
                                        Input(TestFold,:), ...
                                        Output(TestFold,:), ...
                                        randomGuy, ...
                                        true,  ...
                                        regression ...
                                     );
            LCurves(index+1) = LC(end);
            VCurves(index+1) = VC(end);
        end
        
        meanError = mean(VCurves,1);
        
        if isnan(meanError(end))
           meanError(end) = 1000; 
        end
        
        Errors(RandomSearch) = meanError(end);
        
        if BestError > meanError(end)  % update best configuration of parameters and error
            BestError = meanError(end);
            BestGuy = randomGuy;
        end
        
        if mod(RandomSearch, floor(Limit/10)) == 0
            fprintf (RandomSearch + "# try, Mean Error: " + meanError(end) + ...
                    ", Best so far: " + BestError + ...
                    ", Time elapsed: " + toc + "\n");
        end
        
        if meanError(end) < EarlyStoppingError || RandomSearch == Limit
                fprintf("Fantastic Error Found!\n");
                Random_print(BestGuy,BestError);
                Result = BestGuy;
            return
        end
        
    end        

    