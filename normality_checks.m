function [sk, ku, h, p] = normality_checks(x, label)

sk = skewness(x); ku = kurtosis(x);

[h,p] = lillietest(x);

fprintf('\n %s: skew=%.3f | kurt=%.3f | Lillie H=%d | p=%.4g\n', label, sk, ku, h, p);

tiledlayout(1,2,'TileSpacing','loose','Padding','loose');

% Histogram
nexttile; histogram(x, 'NumBins', 10)
xlabel(label); ylabel('Frequency'); 
box on

% Q-Q plot
nexttile; qqplot(x); title('') 
xlabel('Theoretical Quantiles (Normal)'); ylabel('Sample Quantiles')
box on

% Set figure size for consistent export
set(gcf,'Units','inches','Position',[0 0 11 4])
exportgraphics(gcf,[label ' residuals_QC.png'],'Resolution',300)

end
