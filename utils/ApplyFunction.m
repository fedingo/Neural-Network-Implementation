function [result] = ApplyFunction (func, matrix_array, optional)


array_size = size(matrix_array,1);
result = cell(array_size);

if nargin <= 2
    for i = 1:array_size
        result{i} = func(matrix_array{i});
    end
else
    for i = 1:array_size
        result{i} = func(matrix_array{i}, optional{i});
    end
end