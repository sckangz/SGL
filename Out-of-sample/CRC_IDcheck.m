%% **************
% the code is provided to repduce our result reported in our paper [A2]. NOTE THAT [A2] is a journal extension of our AAAI'15 paper [A1].

% [A1] Xi Peng, Zhang Yi, and Huajin Tang,
%      Robust Subspace Clustering via Thresholding Ridge Regression,
%      The Twenty-Ninth AAAI Conference on Artificial Intelligence (AAAI), Austin, Texas, USA, January 25–29, 2015.
% [A2]Xi Peng, Huajin Tang, Lei Zhang, Zhang Yi, and Shijie Xiao,
%     A Unified Framework for Representation-based Subspace Clustering of Out-of-sample and Large-scale Data,
%     IEEE Trans. Neural Networks and Learning Systems, accepted.

% If the codes or data sets are helpful to you, please appropriately CITE our works. Thank you very much!

% More materials can be found from my website:
%            www.pengxi.me,
%     email: pangsaai [at] gmail [dot] com

% This file is the code of CRC [A3],i.e.,

% [A3] Zhang, Lei, Meng Yang, and Xiangchu Feng.
%      "Sparse representation or collaborative representation: Which helps face recognition?."
%       IEEE International Conference on Computer Vision (ICCV), 2011.
%% **************



function [id]= CRC_IDcheck(D,class_pinv_M,y,Dlabels)
%------------------------------------------------------------------------
% CRC_RLS classification function
coef         =  class_pinv_M*y;
for ci = 1:max(Dlabels)
    coef_c   =  coef(Dlabels==ci);
    Dc       =  D(:,Dlabels==ci);
%     error(ci) = norm(y-Dc*coef_c,2)^2/sum(coef_c.*coef_c);
    error(ci) = norm(y-Dc*coef_c,2)^2;

end

index      =  find(error==min(error));
id         =  index(1);
