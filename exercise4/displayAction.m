function displayMatrix = displayAction(choice)
    for i = 1:3
        for j = 1:4
            if (choice((i-1)*4+j) == 1)
                displayMatrix(i,j) = "L";
            elseif (choice((i-1)*4+j) == 2)
                displayMatrix(i,j) = "R";
            elseif (choice((i-1)*4+j) == 3)
                displayMatrix(i,j) = "U";
            elseif (choice((i-1)*4+j) == 4)
                displayMatrix(i,j) = "D";
            end
        end
    end
end

