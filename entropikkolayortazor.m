%  function [kolayx,kolayy,ortax,ortay,zorx,zory]=entropikkolayortazor(training_x,training_y)
%%
clear all;
rand('seed', 1);
[x,y]=arffoku('C:\Users\Melike Nur Mermer\Desktop\36uci\36uci\primary-tumor.arff');
[ornek,ozellik]=size(x);
sinifsay=max(y);
[eorn,torn]=crossval(ornek,4);
edata=eorn{1,1};
tdata=torn{1,1};
%%
[training_x,training_y,testing_x,testing_y]=egitimtestsetleri(edata,tdata,x,y);

eornsay=length(training_y);
sinifsay=max(training_y);

model=fitcknn(training_x,training_y,'NumNeighbors',7,'Standardize',1);
[label,score,cost] = predict(model,training_x);
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
    for j=1:sinifsay
        if(score(i,j)==1)
            komsuluk(i,j)=7;
        else if(score(i,j)>0.85)
            komsuluk(i,j)=6;
        else if(score(i,j)>0.7)
            komsuluk(i,j)=5;
        else if(score(i,j)>0.57)
            komsuluk(i,j)=4;
        else if(score(i,j)>0.41)
            komsuluk(i,j)=3;
        else if(score(i,j)>0.28)
            komsuluk(i,j)=2;
        else if(score(i,j)>0.14)
            komsuluk(i,j)=1;
            else komsuluk(i,j)=0;
            end
            end
            end
            end
            end
            end
        end
    end
end

entropi=zeros(1,eornsay);
for i=1:eornsay
    for j=1:sinifsay
        if komsuluk(i,j)~=0
        entropi(i)=entropi(i)+(-1)*(komsuluk(i,j)/7)*log2(komsuluk(i,j)/7);
        end
    end
end

minentropi=min(entropi);
maxentropi=max(entropi);
esik1=minentropi+(maxentropi-minentropi)/3;
esik2=minentropi+2*(maxentropi-minentropi)/3;

for i=1:eornsay
        if entropi(i)<esik1
            kolayx(a,:)=training_x(i,:);
            kolayy(a,1)=training_y(i);
            a=a+1;
        else if entropi(i)<esik2               
            orta1x(b,:)=training_x(i,:);
            orta1y(b,1)=training_y(i);
            b=b+1;
        else
            zorx(c,:)=training_x(i,:);
            zory(c,1)=training_y(i);
            c=c+1;
            end
        end
end

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