function [hata,hatabas]=klasiksinirli(klasiknet,maxepoch,training_x,training_y,testing_x,testing_y)
% clear all;
% rng(1);
% [x,y]=arffoku('C:\Users\Melike Nur Mermer\Desktop\36uci\36uci\waveform.arff');
% [ornek,ozellik]=size(x);
% sinifsay=max(y);
% [eorn,torn]=crossval(ornek,4);
% edata=eorn{1,1};
% tdata=torn{1,1};
% [training_x,training_y,testing_x,testing_y]=egitimtestsetleri(edata,tdata,x,y);
% tr_x=transpose(training_x);
% tr_y=transpose(training_y);
% klasiknet = feedforwardnet([10,5,sinifsay]);
% klasiknet = configure(klasiknet,tr_x,tr_y);
%%
% rng(1);
nofclass=max(training_y);
eornsay=length(training_y);
tornsay=length(testing_y);
sinifsay=max(training_y);   
tr_x=transpose(training_x);
tr_y=transpose(training_y);
ts_x=transpose(testing_x);

iw=klasiknet.IW{1};
%örnekler sabit epoch gösterildiðinde curriculumda fark görülüyor 
%klasiknet.trainParam.epochs=5;
klasiknet.trainFcn='trainr';%klasiknet.trainFcn='trainc';
klasiknet.trainParam.showWindow = false;
klasiknet.trainParam.epochs=maxepoch;
klasiknet.trainParam.goal=0.01;
%klasiknet.trainParam.max_fail=10;
klasiktahmin=round(klasiknet(ts_x));
[noffeature,nofsample]=size(ts_x);
klasikbasarili=0;
for i=1:nofsample
    if klasiktahmin(i)==testing_y(i);
    klasikbasarili=klasikbasarili+1;
    end
end
hatabas=1-klasikbasarili/nofsample;
% if(tr_y)
% [klasiknet,tr] = train(klasiknet,tr_x,tr_y);
% end
% klasiktahmin=round(klasiknet(ts_x));
% [noffeature,nofsample]=size(ts_x);
% klasikbasarili=0;
% for i=1:nofsample
%     if klasiktahmin(i)==testing_y(i);
%     klasikbasarili=klasikbasarili+1;
%     end
% end
% hatakl(1)=1-klasikbasarili/nofsample;
% 
% if(tr_y)
% [klasiknet,tr] = train(klasiknet,tr_x,tr_y);
% end
% klasiktahmin=round(klasiknet(ts_x));
% [noffeature,nofsample]=size(ts_x);
% klasikbasarili=0;
% for i=1:nofsample
%     if klasiktahmin(i)==testing_y(i);
%     klasikbasarili=klasikbasarili+1;
%     end
% end
% hatakl(2)=1-klasikbasarili/nofsample;

if(tr_y)
[klasiknet,tr] = train(klasiknet,tr_x,tr_y);
end
klasiktahmin=round(klasiknet(ts_x));
[noffeature,nofsample]=size(ts_x);
klasikbasarili=0;
for i=1:nofsample
    if klasiktahmin(i)==testing_y(i);
    klasikbasarili=klasikbasarili+1;
    end
end
hata=1-klasikbasarili/nofsample;