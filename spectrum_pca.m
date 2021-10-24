function spectrum = spectrum_pca(hsi_data, dim)
    [~,score,~,~,~,~] = pca(hsi_data);
    spectrum = score(:, 1:dim);
end