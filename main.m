clear all;
rand('seed', 1);
datasetler={'labor'};
tekrarsayi=5;
cvfold=4;
algsayi=6;
datasetsayi=1;
matfile=zeros(cvfold*tekrarsayi,algsayi,datasetsayi);
tahmin=zeros(cvfold*tekrarsayi,algsayi);
for t=1:datasetsayi
dizin='C:\Users\Melike Nur Mermer\Desktop\36uci\36uci\';
dsdizin=strcat(dizin,datasetler{t},'.arff');
[x,y]=arffoku(dsdizin);
[ornek ,ozellik]=size(x);
sinifsay=max(y);
i=1;
while i<=cvfold*tekrarsayi
[eorn,torn]=crossval(ornek,cvfold);
for j=1:cvfold
edata=eorn{1,j};
tdata=torn{1,j};
[training_x,training_y,testing_x,testing_y]=egitimtestsetleri(edata,tdata,x,y);

%  tumdatax=[training_x; testing_x];
%  tumdatay=[training_y; testing_y];
%  [kolaysayisix,kolaysayisiy]=sadecekolayzor(tumdatax,tumdatay,1);
% 
% [sadecekolaydatasetx,sadecekolaydatasety]=sadecekolayzor(training_x,training_y,1);
% [sadecezordatasetx,sadecezordatasety]=sadecekolayzor(training_x,training_y,2);

klasiknet = feedforwardnet([10,5,sinifsay]);
tr_x=transpose(training_x);
tr_y=transpose(training_y);
klasiknet = configure(klasiknet,tr_x,tr_y);
[tahmin(i,1), epochs(i)]=klasik(klasiknet,training_x,training_y,testing_x,testing_y);
i=i+1;
end
end
numepochs=round(sum(epochs)/(cvfold*tekrarsayi));
% dizin='C:\Users\Melike Nur Mermer\Desktop\36uci\36uci\';
% dsdizin=strcat(dizin,datasetler{t},'.arff');
% [x,y]=arffoku(dsdizin);
% [ornek ,ozellik]=size(x);
% sinifsay=max(y);
i=1;
while i<=cvfold*tekrarsayi
[eorn,torn]=crossval(ornek,cvfold);
for j=1:cvfold
edata=eorn{1,j};
tdata=torn{1,j};
[training_x,training_y,testing_x,testing_y]=egitimtestsetleri(edata,tdata,x,y);

%  tumdatax=[training_x; testing_x];
%  tumdatay=[training_y; testing_y];
%  [kolaysayisix,kolaysayisiy]=sadecekolayzor(tumdatax,tumdatay,1);
% 
% [sadecekolaydatasetx,sadecekolaydatasety]=sadecekolayzor(training_x,training_y,1);
% [sadecezordatasetx,sadecezordatasety]=sadecekolayzor(training_x,training_y,2);

klasiknet = feedforwardnet([10,5,sinifsay]);
tr_x=transpose(training_x);
tr_y=transpose(training_y);
klasiknet = configure(klasiknet,tr_x,tr_y);
net1=klasiknet;
net2=klasiknet;
net3=klasiknet;
net4=klasiknet;
net5=klasiknet;
net6=klasiknet;

% [tahmin(i,1),hatabas(i,:)]=klasiksinirli(net1,round(numepochs),training_x,training_y,testing_x,testing_y);
[tahmin(i,2),hatabas(i,:)]=klasiksinirli(net2,2*round(numepochs),training_x,training_y,testing_x,testing_y);
[tahmin(i,3),hatacr(i,:)]=incremental_planli(net3,round(numepochs/3),training_x,training_y,testing_x,testing_y);
[tahmin(i,4),hatacr(i,:)]=incremental_planli(net4,2*round(numepochs/3),training_x,training_y,testing_x,testing_y);
[tahmin(i,5),hataacr(i,:)]=incremental_tersplanli(net5,round(numepochs/3),training_x,training_y,testing_x,testing_y);
[tahmin(i,6),hataacr(i,:)]=incremental_tersplanli(net6,2*round(numepochs/3),training_x,training_y,testing_x,testing_y);
% tahmin(i,4)=incremental_tersplanli(net4,2*round(numepochs/3),training_x,training_y,testing_x,testing_y);

i=i+1;
end
end
matfile(:,:,t)=tahmin;
save matfile.mat matfile;

for i=1:algsayi-1
    for j=i+1:algsayi
    H(i,j)=ttest2(tahmin(:,i),tahmin(:,j));
    end
end
ttestmat(:,:,t)=H;
save ttestmat.mat ttestmat;
end
% ortalamahata=zeros(1,algsayi);
% for i=1:algsayi
% ortalamahata(1,i)=sum(tahmin(:,i))/(cvfold*tekrarsayi);
% end

% for i=1:3
% orthatacr(i)=sum(hatacr(:,i))/(cvfold*tekrarsayi);
% orthataacr(i)=sum(hataacr(:,i))/(cvfold*tekrarsayi);
% end
% orthatakl(1)=sum(tahmin(:,5))/(cvfold*tekrarsayi);
% orthatakl(2)=sum(tahmin(:,6))/(cvfold*tekrarsayi);
% orthatakl(3)=sum(tahmin(:,2))/(cvfold*tekrarsayi);
% 
% orthatabas=sum(hatabas(:,1))/(cvfold*tekrarsayi);
% 
% aralardaortalamahata(:,1)=transpose(orthatakl);
% aralardaortalamahata(:,2)=transpose(orthatacr);
% aralardaortalamahata(:,3)=transpose(orthataacr);

% H1=ttest2(tahmin(:,1),tahmin(:,2));
% H2=ttest2(tahmin(:,1),tahmin(:,3));
% plansýz-plansýz(n/3)-planlý-tersplanlý
% H=zeros(algsayi);

% 
% % n/2 epoch sýnýrý verilen yöntemler
% H2=zeros(4);
% for i=1:3
%     for j=i+1:4
%     H2(i,j)=ttest2(hatalar(:,i),hatalar(:,j));
%     end
% end
