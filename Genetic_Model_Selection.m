function [result] = Genetic_Model_Selection (Input, Output, K, Ranges, regression)

% K = number of folds for Cross-validation
% Ranges = ranges where to search for each hyper-parameter
% Regression = boolean variable

if nargin < 5
    regression = false;
end

% Parameters for the Genetic Algorithm
Population_Size = 100;
Generations = 10;
RestartLimit = 500;
MutationRate = 0.02;
ParentsRate = 0.1;
EarlyStoppingError = 0.01;


Generation = cell(Population_Size,1);
BestGuyError = 100;    % best error found up to now
BestGuy = cell(0,0);   % best configuration of parameters up to now

%Initialization
for i = 1:Population_Size
    Generation{i} = GenerateParameters(Ranges); % generate random parameters
end

fprintf("Starting... \n");
tic;

for Gen = 1:Generations
    weights = zeros(Population_Size,1);
    Errors = zeros(Population_Size,1);
    
    for i = 1:Population_Size
        
        Errors(i) = CrossValidation(Input, Output, K, Generation{i}, regression);
        weights(i) = 1/(Errors(i) + 0.01);  % higher errors lead to lower weights
        
        if isnan(Errors(i)) % Consistency Check
            Errors(i) = 1000;
        end
        
        % Early Stop Checking
        if Errors(i) < EarlyStoppingError
            fprintf( "Good Error Found!\n");
            Random_print(Generation{i},Errors(i));

            result = Generation{i};
            return;
        end
    end
    
    fprintf( "Generation " + Gen + ": "+ ...
             "Min: " + min(Errors) + ", "+...
             "Mean: " + mean(Errors) + ", " +...
             "Time Taken: " + toc + "\n" ...
         );
    
    [~,i] = min(Errors);
    
    if BestGuyError > Errors(i)
        BestGuyError = Errors(i);
        BestGuy =  Generation{i};
    end
     
    weights = weights - min(weights) + 0.1; % rescaling of the weights
    
    newGeneration = cell(Population_Size,1);
    
    % If the error is grater than the Restart Limit we re-Randomize
    % all the population
    
    if Errors(i) > RestartLimit
        %Re-Randomization
        fprintf ("Errors in Generation are too high, restarting...\n");
        
        for i = 1:Population_Size         
            newGeneration{i} = GenerateParameters(Ranges);
        end
        
    else
        % Otherwise we create the New_Generation by combining the best
        % parents of the Old_Generation
        parents = 0;
        
        % PARENTS CHOICE
        while size(parents,1) < 3
            % select the parents as the ParentsRate percentage of the
            % population based on weights
            parents = randsample(Population_Size,Population_Size*ParentsRate, ...
                true,weights);
            parents = unique(parents);
        end

        % new population generation
        for i = 1:Population_Size
           parentsIndex = randi( size(parents,1), 2,1);

           mum = Generation{parents(parentsIndex(1))};
           dad = Generation{parents(parentsIndex(2))};
           
           mySize = size(mum,1);

           newGenome = cell(mySize,1); 
           
           for j = 1:mySize
               if rand() < MutationRate %Mutation
                    
                   % Generate Random Genome and take the appropriate one
                   mutator = GenerateParameters(Ranges);
                   newGenome{j} = mutator{j};
                   
               %Crossover
               elseif rand() > 0.5 
                   newGenome(j) = mum(j);
               else
                   newGenome(j) = dad(j);
               end
           end
           
           newGeneration{i} = newGenome; % new child

        end
    end
    
    Generation = newGeneration;
end

% Last evaluation of the population
finals = zeros(Population_Size,1);

for i = 1:Population_Size
        finals(i) = CrossValidation(Input, Output, K, Generation{i}, regression);
end

[~, i] = min(finals);

if BestGuyError > finals(i)
    BestGuyError = finals(i);
    BestGuy = Generation{i};
end

Random_print(BestGuy,BestGuyError);
result = BestGuy;

