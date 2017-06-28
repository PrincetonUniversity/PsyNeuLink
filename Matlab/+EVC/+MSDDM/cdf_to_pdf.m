function [tArrayPDF,finalPDF] = cdf_to_pdf(tArray,inputCDF)

for jj = 1:length(inputCDF)-1
    yy(jj) = (inputCDF(jj+1)-inputCDF(jj))/(tArray(jj+1)-tArray(jj));
end

 tArrayPDF = tArray(2:end);

 finalPDF = yy/trapz(tArrayPDF,yy);

%  finalPDF