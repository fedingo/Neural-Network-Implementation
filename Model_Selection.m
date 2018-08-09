function Model_Selection (Input, Output, K, Ranges, Regression)

    % K = number of folds for Cross-validation
    % Ranges is an array of 4x2
    % Regression = boolean variable
        
    % Number of Input samples and Output Samples must be equal
    if size(Input,1) ~= size(Output,1)
        fprintf("Input and Output are not of the same size!\n");
        return;
    end
    
    % Our Neural Network
    % Hyperparameters:
    %   -Eta
    %   -Lambda
    %   -Alpha
    %   -Number of units for one hidden layers (We start from this)
    
    NumberOfSamples = size(Input,1);
    FoldSize = floor(NumberOfSamples/K);
    
    shuffle = randperm(NumberOfSamples);
    
    LCurves = cell(4,4); % Matrix of the Learning Curves
    VCurves = cell(4,4); % Matrix of the Validation Curves
        
    for index = 0:K-1
        tic;
        % Training fold = difference between the all dataset shuffled and the TestFold
        TestFold = shuffle(FoldSize*index+1:FoldSize*(index+1));
        TrainingFold = setdiff(shuffle, TestFold);

        % Grid Search
        for hidden = 1:2
        for eta = 1:2
        for lambda = 1:2
        for alpha = 1:2
            [LC, VC] = Training(  Input(TrainingFold,:), ...
                                        Output(TrainingFold,:), ...
                                        Input(TestFold,:), ...
                                        Output(TestFold,:), ...
                                        {Ranges(1,hidden), ...
                                        Ranges(2,eta), ...
                                        Ranges(3,lambda), ...
                                        Ranges(4,alpha)},  ...
                                        false, ...
                                        Regression ...
                                     );
            LCurves{hidden*2 + eta-2, lambda*2 + alpha-2} = ...
                [ LCurves{hidden*2 + eta-2, lambda*2 + alpha-2} LC];
            VCurves{hidden*2 + eta-2, lambda*2 + alpha-2} = ...
                [ VCurves{hidden*2 + eta-2, lambda*2 + alpha-2} VC];
        end
        end
        end
        end
        
        fprintf("Time for the "+(index+1)+"/"+K+" Fold: "+ toc + "\n");
    end
    
    k=1;
    id = 0;    
    
    for hidden = 1:2
    for eta = 1:2
    for lambda = 1:2
    for alpha = 1:2      
        subplot(4,4,k)
        k = k+1;
        hold on;
        for i = 1:K
            plot(LCurves{hidden*2 + eta-2, lambda*2 + alpha-2}(:,i),...
                    "Color",[1,0.2,0.2],'LineWidth',0.5)
            plot(VCurves{hidden*2 + eta-2, lambda*2 + alpha-2}(:,i),...
                    "Color",[0.2,0.2,1],'LineWidth',0.5)
        end

        plot(mean(LCurves{hidden*2 + eta-2, lambda*2 + alpha-2},2),...
                    "red", 'LineWidth',2);
        meanValidationCurve = mean(VCurves{hidden*2 + eta-2, lambda*2 + alpha-2},2);
        plot(meanValidationCurve,...
                    "blue", 'LineWidth',2);
        hold off;
        
        id = id + 1;
        precision = 2;
        
        title(  "ID=" + id + ", " + ...
                "H="  + Ranges(1, hidden) + ", "+...
                "E="  + num2str(Ranges(2, eta),precision)    + ", "+...
                "L="  + num2str(Ranges(3, lambda),precision) + ", "+...
                "A="  + num2str(Ranges(4, alpha),precision)  + ", "+...
                "LE=" + num2str(meanValidationCurve(end),precision+1) )
    end
    end
    end
    end

    
    