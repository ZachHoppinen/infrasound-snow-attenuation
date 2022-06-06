function [corrs,samples,timelags,P]=correlofast(trace1,trace2,wl,ol,deltmax,sps)

% correlofast.m performs a running correlation calculation on two time
% series and returns a matrix of normalized cross correlation values.  It
% peforms correlations in the frequency domain and is much faster than
% correplot.  Input data vectors must be time aligned.  This script is
% optimal for processing long time series
%
% USAGE:
%[corrs,samples,timelag]=correlofast(trace1,trace2,window,overlap,deltmax,resize)
%   
% INPUT VARIABLES:
%       trace1 and trace2 (input time series to be compared)
%       wl (time length in samples of the running comparison window...
%           if no window length is declared the correlation will be applied
%           to the entire time series)
%       ol (overlap of comparison window... default is 1/2 window)
%       deltmax (+/-range of lag times that are searched... optional)
%       resize (increase the time resolution of the cross-correlation... optional)
%       normalize ('coeff' or 'none')
%
% OUTPUT VARIABLES
%       corrs (matrix of cross correlations aranged as columns)
%       samples (middle sample of the correlation window)
%       timelag (delay times associated with cross correlation values)
%       traces1/traces2 (optional output with reshaped matrix of all time
%       series)
%       P (power associated with each cross-correlation time window)
%
%
% EXAMPLE:
%
% AUTHOR: Jeff Johnson (jeffreybjohnson@boisestate.edu) updated 6/11/21

if nargin<6
    sps = 100;
end
sps;
ol;
wl;
time_step = wl-ol; 
num_windows = floor(length(trace1)/time_step);
N = time_step*num_windows; % truncate samples that are not evenly divisible by the time step


TRACE1tmp = reshape(trace1(1:N),time_step,num_windows);
TRACE2tmp = reshape(trace2(1:N),time_step,num_windows);

TRACE1 = TRACE1tmp;
TRACE2 = TRACE2tmp;

% this loop is used to create overlapping windows
if round(wl/time_step) ~= wl/time_step
    disp('window length must be an integer multiple of time steps... exiting')
    return
end
for k=2:wl/time_step
    TRACE1 = [TRACE1; circshift(TRACE1tmp,-k+1,2)];
    TRACE2 = [TRACE2; circshift(TRACE2tmp,-k+1,2)];
end

% window overlapping traces with Hanning window
HANN = repmat(hann(wl),1,num_windows);
T1 = TRACE1.*HANN;
T2 = TRACE2.*HANN;

% calculate frequency domain equivalents of time series
FFT1 = fft(T1);
FFT2 = fft(T2);

% normalization factor
NORM21 = sqrt(repmat(sum(abs(FFT1).^2).*sum(abs(FFT2).^2),wl,1));

% create matrix of overlapping correlation functions
if nargin>6 % don't normalize!
    CORRS21 = wl*fftshift(ifft(FFT1.*conj(FFT2)),1);
else
    CORRS21 = wl*fftshift(ifft(FFT1.*conj(FFT2)./NORM21),1);
end

% create other output values
samples = (0:num_windows-1)*time_step + wl/2; % sample index for center of each window
timelags = (1:size(CORRS21,1)) - wl/2 - 1; % time lags in samples
P = NORM21(1,:)/wl; % this is the power of the cross correlation trace

corrs = CORRS21;

% if maximum time lag is declared crop both the correlation matrix and the  
if nargin>4
    lag_indices = abs(timelags)<=deltmax;
    timelags = timelags(lag_indices);
    corrs = CORRS21(lag_indices,:);
end

disp(['processing ' num2str(num_windows) ' windows of ' num2str(wl) ' samples each; overlap is ' num2str(ol/wl*100) '%'])

% new below
trash = samples>(length(trace1)-samples(1));
disp(['getting rid of the last ' num2str(sum(trash)) ' samples'])
P(:,trash) = [];
corrs(:,trash) = [];
samples(:,trash) = [];


