function [pattern] = GenerateParameters(range)

% Generation of the Hyperparameters for a training
    
    % first range is [max_layers, max_neurons]
    number_of_layers = randi( range(1,1) );

    % Log normalization of Learning Rate
    range(2,:) = log(range(2,:))./log(10);
    
    % Hyperparameters range Length added
    range = [range (range(:,2)-range(:,1))];
    
    pattern = cell(4,1);
    
    for j = 1:4
        if j == 1
            Layers = zeros(number_of_layers,1);
            for L = 1:number_of_layers
                Layers(L) = randi(range(1,2)-1) + 1; 
                % min_neurons is chosen to be 2
            end
            pattern{1} = Layers;
        elseif j == 2 %eta
            pattern{j} = 10^(range(j,3) * rand() + range(j,1));
        else % lamba, alfa
            pattern{j} = range(j,3) * rand() + range(j,1);
        end
    end