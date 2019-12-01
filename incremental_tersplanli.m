function [hata,hataacr]=incremental_tersplanli(net,maxepoch,training_x,training_y,testing_x,testing_y)
%%
% clear all;
% rand('seed', 1);
% [x,y]=arffoku('C:\Users\Melike Nur Mermer\Desktop\36uci\36uci\segment.arff');
% [ornek,ozellik]=size(x);
% sinifsay=max(y);
% [eorn,torn]=crossval(ornek,4);
% edata=eorn{1,1};
% tdata=torn{1,1};
% [training_x,training_y,testing_x,testing_y]=egitimtestsetleri(edata,tdata,x,y);
% 
% tr_x=transpose(training_x);
% tr_y=transpose(training_y);
% ts_x=transpose(testing_x);
% 
% net1 = feedforwardnet([10,5,sinifsay]);
% net1 = configure(net1,tr_x,tr_y);
% net=net1;
% net2=net1;
% net3=net1;
% wnet=net.IW{1};
% wnet1=net1.IW{1};
% wnet2=net2.IW{1};
% tahmin = round(net2(ts_x));
% [noffeature,nofsample]=size(ts_x);
% basarili=0;
% for k=1:nofsample
%     if tahmin(k)==testing_y(k);
%     basarili=basarili+1;
%     end
% end
% hatabas=1-basarili/nofsample;
% 
% net1.trainFcn='trainr';
% [net1,tr1] = train(net1,tr_x,tr_y);
% maxepoch=round(tr1.num_epochs/3);
% tahmin = round(net1(ts_x));
% [noffeature,nofsample]=size(ts_x);
% basarili=0;
% for k=1:nofsample
%     if tahmin(k)==testing_y(k);
%     basarili=basarili+1;
%     end
% end
% hata=1-basarili/nofsample;
% 
% net2.trainFcn='trainc';
% net2.trainParam.epochs=maxepoch;
% % net2.trainParam.max_fail=100;
% [net2,tr2] = train(net2,tr_x,tr_y);
% tahmin = round(net2(ts_x));
% [noffeature,nofsample]=size(ts_x);
% basarili=0;
% for k=1:nofsample
%     if tahmin(k)==testing_y(k);
%     basarili=basarili+1;
%     end
% end
% hatakl(1)=1-basarili/nofsample;
% [net2,tr2] = train(net2,tr_x,tr_y);
% tahmin = round(net2(ts_x));
% [noffeature,nofsample]=size(ts_x);
% basarili=0;
% for k=1:nofsample
%     if tahmin(k)==testing_y(k);
%     basarili=basarili+1;
%     end
% end
% hatakl(2)=1-basarili/nofsample;
% [net2,tr2] = train(net2,tr_x,tr_y);
% tahmin = round(net2(ts_x));
% [noffeature,nofsample]=size(ts_x);
% basarili=0;
% for k=1:nofsample
%     if tahmin(k)==testing_y(k);
%     basarili=basarili+1;
%     end
% end
% hatakl(3)=1-basarili/nofsample;
%%
% rng(1);
[kolayx,kolayy,ortax,ortay,zorx,zory]=entropikensemblekolayortazor(training_x,training_y);
ts_x=transpose(testing_x);

net.trainFcn='trainc';
net.trainParam.showWindow = false;
net.trainParam.epochs=maxepoch;
% net.trainParam.goal=0.01;
%net.trainParam.max_fail=10;
%net.trainParam.epochs=round(length(kolayy));
%net.trainParam.goal=0.001;
if(zory)
[net,tr] = train(net,zorx,zory);
iw=net.IW{1};
%net.trainParam.goal=min(tr.perf);
%net.trainParam.epochs=round(length(ortay));
end
[noffeature,nofsample]=size(ts_x);
tahmin = round(net(ts_x));
basarili=0;
for k=1:nofsample
    if tahmin(k)==testing_y(k);
    basarili=basarili+1;
    end
end
hataacr(1)=1-basarili/nofsample;
if(ortay)
%orta örnekler de maxepoch defa gösterilecek
%[net,y1,e1,pf] = adapt(net,ortax,ortay);%1 defa burda gösterildi
%for i=1:maxepoch-1%orta örnekler de belirlenen iterasyon sayýsý kadar gösteriliyor
%wo(:,:,i)=net.IW{1};
[net,tr] = train(net,ortax,ortay);
iw2=net.IW{1};
%[net,y1,e1,pf] = adapt(net,ortax,ortay);
%end
end
[noffeature,nofsample]=size(ts_x);
tahmin = round(net(ts_x));
basarili=0;
for k=1:nofsample
    if tahmin(k)==testing_y(k);
    basarili=basarili+1;
    end
end
hataacr(2)=1-basarili/nofsample;
%net.trainParam.epochs=round(length(zory));
if(kolayy)
%[net,y1,e1,pf] = adapt(net,zorx,zory);
%for j=1:maxepoch-1%zor örnekler de belirlenen iterasyon sayýsý kadar gösteriliyor
% wz(:,:,j)=net.IW{1};
%[net,tr] = train(net,zorx,zory);
[net,tr] = train(net,kolayx,kolayy);
iw3=net.IW{1};
%[net,y1,e1,pf] = adapt(net,zorx,zory);
%end
end

% if(ortay)
% [net,tr] = train(net,ortax,ortay);
% end
% [net,tr] = train(net,kolayx,kolayy);

%netin örnekleri tahmin etmesi

tahmin = round(net(ts_x));

[noffeature,nofsample]=size(ts_x);
basarili=0;
for k=1:nofsample
    if tahmin(k)==testing_y(k);
    basarili=basarili+1;
    end
end
hata=1-basarili/nofsample;
hataacr(3)=hata;