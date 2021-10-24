function averaged_hsi_data = spectrum_average(hsi_data)
    n = size(hsi_data, 2);
    step_size = 16;
    averaged_hsi_data = zeros(size(hsi_data, 1), 256);
    current_average_index = 1;
    for i = 1:step_size:n
        if (i + step_size < n)
            s = mean(hsi_data(:, i:i+step_size), 2);
            averaged_hsi_data(:, current_average_index) = s;
        end
        current_average_index = current_average_index + 1;
        if step_size == 16
            step_size = 17;
        else
            step_size = 16;
        end
    end
end
