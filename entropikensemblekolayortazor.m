function [kolayx,kolayy,ortax,ortay,zorx,zory]=entropikensemblekolayortazor(training_x,training_y)
%%
% clear all;
% rand('seed', 1);
% [x,y]=arffoku('C:\Users\Melike Nur Mermer\Desktop\36uci\36uci\primary-tumor.arff');
% [ornek,ozellik]=size(x);
% sinifsay=max(y);
% [eorn,torn]=crossval(ornek,4);
% edata=eorn{1,1};
% tdata=torn{1,1};
% [training_x,training_y,testing_x,testing_y]=egitimtestsetleri(edata,tdata,x,y);

[eornsay,ozellik]=size(training_x);
sinifsay=max(training_y);

ensmsay=10;%10 aðaçlý ensemble
bagger=randi(eornsay,ensmsay,eornsay);
to_x=zeros(eornsay,ozellik,ensmsay);
to_y=zeros(eornsay,1,ensmsay);    
        for i=1:ensmsay
        for j=1:eornsay
        to_x(j,:,i)=training_x(bagger(i,j),:);
        to_y(j,1,i)=training_y(bagger(i,j),1);
        end
        end
%döngü içinde aðaç oluþturabilmek için "dtrees" cell array oluþturulur.
dtrees = cell(1,ensmsay);       
for i=1:ensmsay
tree=fitctree(to_x(:,:,i),to_y(:,1,i),'Prune','off','MinLeafSize',1,'MinParentSize',2);
dtrees{i} = tree;
end
to_tahmin=zeros(eornsay,1,ensmsay);
olasilik=zeros(eornsay,sinifsay);
for i=1:ensmsay
tree=dtrees{i};
[to_tahmin(:,1,i),score,depth,cnum] = predict(tree,training_x);
end

for i=1:ensmsay
for j=1:eornsay
    olasilik(j,to_tahmin(j,1,i))=olasilik(j,to_tahmin(j,1,i))+1;
end
end

entropi=zeros(1,eornsay);
for i=1:eornsay
    for j=1:sinifsay
        if olasilik(i,j)~=0
        entropi(i)=entropi(i)+(-1)*(olasilik(i,j)/ensmsay)*log2(olasilik(i,j)/ensmsay);
        end
    end
end

minentropi=min(entropi);
maxentropi=max(entropi);
esik1=minentropi+(maxentropi-minentropi)/3;
esik2=minentropi+2*(maxentropi-minentropi)/3;

a=1;
b=1;
c=1;
%m=1;
kolayx=[];
orta1x=[];
zorx=[];
kolayy=[];
orta1y=[];
zory=[];

for i=1:eornsay
        if entropi(i)<esik1
            kolayx(a,:)=training_x(i,:);
            kolayy(a,1)=training_y(i);
            kolayentropi(a)=entropi(i);
            a=a+1;
        else if entropi(i)<esik2               
            orta1x(b,:)=training_x(i,:);
            orta1y(b,1)=training_y(i);
            ortaentropi(b)=entropi(i);
            b=b+1;
        else
            zorx(c,:)=training_x(i,:);
            zory(c,1)=training_y(i);
            zorentropi(c)=entropi(i);
            c=c+1;
            end
        end
end

% kolayx=[zorx;orta1x;kolayx];
% kolayy=[zory;orta1y;kolayy];
% 
% ortax=[zorx;orta1x;kolayx];
% ortay=[zory;orta1y;kolayy];
entropi=transpose(entropi);

% histogram(kolayentropi,'BinWidth', 0.1000);
% hold on;
% histogram(ortaentropi,'BinWidth', 0.1000);
% hold on;
% histogram(zorentropi,'BinWidth', 0.1000);

ortax=[orta1x;orta1x;kolayx];
ortay=[orta1y;orta1y;kolayy];

zorx=[zorx;zorx;zorx;orta1x;kolayx];
zory=[zory;zory;zory;orta1y;kolayy];

kolayx=transpose(kolayx);
ortax=transpose(ortax);
zorx=transpose(zorx);
kolayy=transpose(kolayy);
ortay=transpose(ortay);
zory=transpose(zory);