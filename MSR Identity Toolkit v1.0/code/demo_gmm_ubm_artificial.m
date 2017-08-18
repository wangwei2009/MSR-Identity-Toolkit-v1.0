%{

This is a demo on how to use the Identity Toolbox for GMM-UBM based speaker
recognition. A small scale task has been designed using artificially
generated features for 20 speakers. Each speaker has 10 sessions
(channels) and each session is 1000 frames long (10 seconds assuming 10 ms
frame increments).

There are 4 steps involved:
 
 1. training a UBM from background data
 2. MAP adapting speaker models from the UBM using enrollment data
 3. scoring verification trials
 4. computing the performance measures (e.g., confusion matrix and EER)

Note: given the relatively small size of the task, we can load all the data 
and models into memory. This, however, may not be practical for large scale 
tasks (or on machines with a limited memory). In such cases, the parameters 
should be saved to the disk.

Malcolm Slaney <mslaney@microsoft.com>
Omid Sadjadi <s.omid.sadjadi@gmail.com>
Microsoft Research, Conversational Systems Research Center

%}

% 
%% 
% Step0: Set the parameters of the experiment
nSpeakers = 20;
nDims = 13;             % dimensionality of feature vectors
nMixtures = 32;         % How many mixtures used to generate data
nChannels = 10;         % Number of channels (sessions) per speaker
nFrames = 1000;         % Frames per speaker (10 seconds assuming 100 Hz)
nWorkers = 1;           % Number of parfor workers, if available

% Pick random centers for all the mixtures.
mixtureVariance = .10;
channelVariance = .05;
mixtureCenters = randn(nDims, nMixtures, nSpeakers);
channelCenters = randn(nDims, nMixtures, nSpeakers, nChannels)*.1;
trainSpeakerData = cell(nSpeakers, nChannels);
testSpeakerData = cell(nSpeakers, nChannels);
speakerID = zeros(nSpeakers, nChannels);

% Create the random data. Both training and testing data have the same
% layout.
for s=1:nSpeakers
    trainSpeechData = zeros(nDims, nMixtures);
    testSpeechData = zeros(nDims, nMixtures);
    for c=1:nChannels
        for m=1:nMixtures
            % Create data from mixture m for speaker s
            frameIndices = m:nMixtures:nFrames;
            nMixFrames = length(frameIndices);
            trainSpeechData(:,frameIndices) = ...
                randn(nDims, nMixFrames)*sqrt(mixtureVariance) + ...
                repmat(mixtureCenters(:,m,s),1,nMixFrames) + ...
                repmat(channelCenters(:,m,s,c),1,nMixFrames);
            testSpeechData(:,frameIndices) = ...
                randn(nDims, nMixFrames)*sqrt(mixtureVariance) + ...
                repmat(mixtureCenters(:,m,s),1,nMixFrames) + ...
                repmat(channelCenters(:,m,s,c),1,nMixFrames);
        end
        trainSpeakerData{s, c} = trainSpeechData;
        testSpeakerData{s, c} = testSpeechData;
        speakerID(s,c) = s;                 % Keep track of who this is
    end
end

%%
% Step1: Create the universal background model from all the training speaker data
nmix = nMixtures;           % In this case, we know the # of mixtures needed
final_niter = 10;
ds_factor = 1;
ubm = gmm_em(trainSpeakerData(:), nmix, final_niter, ds_factor, nWorkers);

%%
% Step2: Now adapt the UBM to each speaker to create GMM speaker model.
map_tau = 10.0;
config = 'mwv';
gmm = cell(nSpeakers, 1);
for s=1:nSpeakers
    gmm{s} = mapAdapt(trainSpeakerData(s, :), ubm, map_tau, config);
end

%%
% Step3: Now calculate the score for each model versus each speaker's data.
% Generate a list that tests each model (first column) against all the
% testSpeakerData.
trials = zeros(nSpeakers*nChannels*nSpeakers, 2);
answers = zeros(nSpeakers*nChannels*nSpeakers, 1);
for ix = 1 : nSpeakers,
    b = (ix-1)*nSpeakers*nChannels + 1;
    e = b + nSpeakers*nChannels - 1;
    trials(b:e, :)  = [ix * ones(nSpeakers*nChannels, 1), (1:nSpeakers*nChannels)'];
    answers((ix-1)*nChannels+b : (ix-1)*nChannels+b+nChannels-1) = 1;
end

gmmScores = score_gmm_trials(gmm, reshape(testSpeakerData', nSpeakers*nChannels,1), trials, ubm);

%%
% Step4: Now compute the EER and plot the DET curve and confusion matrix
imagesc(reshape(gmmScores,nSpeakers*nChannels, nSpeakers))
title('Speaker Verification Likelihood (GMM Model)');
ylabel('Test # (Channel x Speaker)'); xlabel('Model #');
colorbar; drawnow; axis xy
figure
eer = compute_eer(gmmScores, answers, true);
