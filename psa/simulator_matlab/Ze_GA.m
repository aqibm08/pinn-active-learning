% function [puritylist recoverylist masslist Wtotal Productivity t2 INOUT xinput]=PSA_13X_GA(tads_,tblow_,tevac_,PINTc,PLc,V0)
function [puritylist,recoverylist,YALL1,YALL2,YALL3,YALL4]=Ze_GA(Yinit,X)
tic
global iteration

iteration=iteration+1;
cin_index = zeros(1,25);
vpress_=0.2;
tads_=X(1);
vblow_=0.05;
PINTc=0.2;
vevac_=1;
PLc=X(2);
vads_=0.5;
V0 = 0.5;
vpress=vpress_/V0;
vads = vads_/V0;
vblow=vblow_/V0;
vevac=vevac_/V0;

ngrid=25;
ncomponent=2;
NEQ=ngrid*(ncomponent-1+ncomponent+2);
YFEED(1)=X(3);
YFEED(2)=1.0-YFEED(1);
FEEDInitial(1,:)=cin_index;
FEEDInitial(2,:)=1-FEEDInitial(1);

%----bed and adsorbent properties----
BL=1.0d0; %(m) Bed Length
Ro=0.324/2.0;  % (m) Column outer radius
Ri=0.289/2.0; %(m)Column inner radius
RP=0.0015/2; %0.0015/2.0; % (m)Paticle radius
BV=0.37; % Bed Voidage (porosity)
PV=0.35; % particle porosity
TORT=3.0; %tortuoisity factor
KZ=0.0903;  %[J/m/s/K] Effective heat conduction coeffecient
Kw=16.0; %[W/m/K] Thermal conductivity of column wall
hi=0; % [J/m^2/s/K] heat transfe coefficient inside the column
ho=0;  %[W/m^2/K] heat transfer coefficient for outside the column
RG=8.314472; %Universal gas constant [m^3*Pa/(K*mol)]
%----
%---operating conditions----
PREF=1.013E05;
PH=1.0*PREF;
PINT=PINTc*PREF;
PL=PLc*PREF;
ATP=0.3; %parameters for exponential pressure term for Pressuirization step,blowdown step and vacuum
T0=298.15; %(K) Temperature
Tatm=T0;
%----

%----fluid and adsorbent properties and constants----
MW(1)=44.01; %(g/mol) Molecular weight of CO2
MW(2)=28.01; %(g/mol) Molecular weight of N2
AVGMW=sum(YFEED(1:ncomponent).*MW(1:ncomponent));
DM=1.6e-5; %1.2955e-5; % (m^2/s) Molecular diffusivity of CO2-N2 mixture @ 298.14K   (0.12955[cm^2/s])
DP=DM/TORT;
ROWG=(PH*AVGMW/1000)/(RG*T0); %[kg/m^3] density of gas
ROWS=937.748616;  %[kg/m^3] density of solid particle IISERP 
ROWW=7800.0; %[kg/m^3] Density of Column
CPG=1010.6; %[J/kg/K]Specific heat capacity of gas mixture  (1.0106[kJ/kg/K])

CPS=1070.0;   %[J/kg/K] Specific heat capacity of adsorbent (0.92[J/g/K])
CPW=502.0;   %[J/kg/K] Specific heat capacity og column wall  0.502[J/g/K]
CPA=CPG;

MIU=1.72e-5; %[kg/m/s] Viscosity **NEW**
KAPPA= 4.0/150.0*RP^2 * (BV^2/(1-BV)^2); %**NEW**
gamma=1.4;
eta=0.72;
gmin1byg=(gamma-1)/gamma;
%----

%----IISERP-MOF dual-site equilibrium parameters----

B0(1)=9.3880523013e-08; %m^3/mol Langmuir constant for CO2
B0(2)=2.549e-07; %m^3/mol Langmuir constant for N2

D0(1)=5.22911980202e-07;
D0(2)=2.549e-07;

U1(1)=-31135.10206;
U1(2)=-11890.66206;

U2(1)=-31135.10206;
U2(2)=-11890.66206;

H1=U1-RG*T0;
H2=U2-RG*T0;

B=B0.*exp(-U1/(RG*T0)) *1;
D=D0.*exp(-U2/(RG*T0)) *1;

qsat1 = 3.298453336;
qsat2 = 1.891959934;
QS1(1)= qsat1*ROWS;
QS1(2)= qsat1*ROWS;
QS2(1)= qsat2*ROWS;
QS2(2)= qsat2*ROWS;
QSref=QS1(2);
%----

%----dimensionless groups----
RATIO1=QS1/QSref;
RATIO2=QS2/QSref;
H=QS1/QSref.*H1+QS2/QSref.*H2;

DL=0.7*(DM*PREF/PH)+0.5*V0*(2.0*RP);
Pe=V0*BL/DL;
Peh=BV*V0*BL*ROWG*CPG/KZ;

GRP1=(1.0-BV)/BV;

SHI=GRP1*RG*T0*QSref/PH;
PI1=Kw/(ROWW*CPW*V0*BL);
PI2=2.0*Ri*hi*BL/(ROWW*CPW*V0*((Ro^2)-(Ri^2)));
PI3=2.0*Ro*ho*BL/(ROWW*CPW*V0*((Ro^2)-(Ri^2)));

PII4=KZ/(BV*V0*BL);
PII5=CPG/RG *PH/T0 *AVGMW/1000;
PII61=GRP1*QSref*(-H(1))/T0;
PII62=GRP1*QSref*(-H(2))/T0;
PII7=GRP1*(-CPA)*QSref*AVGMW/1000;
PII8=2.0*hi*BL/(BV*V0*Ri);

GAMMA=KAPPA*PH/(MIU*V0*BL);
GAMMA2=1.75*(1-BV)/(BV*2*RP)*ROWG *BL/PH*V0^2;
Rey=ROWG*V0*(2*RP)/MIU;
%----

% %----INITIAL CONDITIONS----
% Y((2*ncomponent+1)*ngrid+(1:ngrid))=PL/PH; %PRESS
% %----
% 
% %----INITIAL FLUID PHASE----
% FEED=FEEDInitial;
% %----
% 
% %----INITIAL SOLID PHASE----
% XEQUIB1=(Y((2*ncomponent+1)*ngrid+(1:ngrid))*PH)./(RG*T0).*(B0(1)*exp(-U1(1)./(RG*T0)))*1;
% XEQUIB2=(Y((2*ncomponent+1)*ngrid+(1:ngrid))*PH)./(RG*T0).*(B0(2)*exp(-U1(2)./(RG*T0)))*1;
% XEQUID1=(Y((2*ncomponent+1)*ngrid+(1:ngrid))*PH)./(RG*T0).*(D0(1)*exp(-U2(1)./(RG*T0)))*1;
% XEQUID2=(Y((2*ncomponent+1)*ngrid+(1:ngrid))*PH)./(RG*T0).*(D0(2)*exp(-U2(2)./(RG*T0)))*1;
% SumXEQUI = 1 + XEQUIB1.*FEED(1,:) + XEQUIB2.*FEED(2,:);
% SumXEQUI2 = 1 + XEQUID1.*FEED(1,:) + XEQUID2.*FEED(2,:);
% 
% XE(1,:)=RATIO1(1).*XEQUIB1.*FEED(1,:)./SumXEQUI+RATIO2(1).*XEQUID1.*FEED(1,:)./SumXEQUI2;
% XE(2,:)=RATIO1(2).*XEQUIB2.*FEED(2,:)./SumXEQUI+RATIO2(2).*XEQUID2.*FEED(2,:)./SumXEQUI2;
% %----
% 
% %----INITIALIZING ARRAY "Y"----
% for i=1:ncomponent-1
%     Y((i-1)*ngrid+(1:ngrid))=FEED(i,:);
% end
% 
% for i=1:ncomponent
%     Y((ncomponent-1+i-1)*ngrid+(1:ngrid))=XE(i,:);
% end
% 
% Y((ncomponent-1+ncomponent)*ngrid+(1:ngrid))=1; %Bed T
% Y((ncomponent-1+ncomponent+1)*ngrid+(1:ngrid))=1; %Wall T
% save Yinit_temp.mat Y
Y = Yinit;

Y(6*ngrid+1)=PH/PH; %Pin
Y(6*ngrid+2)=0; %vin
Y(6*ngrid+3)=YFEED(1); %yin
Y(6*ngrid+4)=1; %Tin

Y(6*ngrid+5)=10;Y(6*ngrid+6)=10*YFEED(1);

%t0=BL/V0;
t0=2;
vinit=0;
delz=1/ngrid; Area=0.25*pi*(2*Ri)^2;
YALL=Y; INOUT=0;

tpress_end=200; lpress=0.0; tads=tads_/t0;tblow_end=300; lblow=0.0;
tevac_end=3000; levac=0.0;
ncycle=500;t2all(ncycle,1)=0;
c1=clock;tc1=cputime;
swtch=0;
n=0;
while swtch==0 % Conditon for CSS
    n=n+1;
    
    %-----------------
        lpress=lpress+1;
        Y=YALL(end,1:6*ngrid);
        Y(6*ngrid+1)=PL/PH;Y(6*ngrid+2)=0;Y(6*ngrid+3)=YFEED(1); Y(6*ngrid+4)=1;
        Y(6*ngrid+5)=10;Y(6*ngrid+6)=10*YFEED(1);
        vinit=-GAMMA*2.0/delz* (YALL(end,6*ngrid+1)-YALL(end,5*ngrid+1));vinit=0;
        delt=0.01;
        options=odeset('Events',@events,'RelTol', [1e-5], 'AbsTol', [1e-5], 'Vectorized','on','JPattern',jpattern1(ngrid));
        t1=cputime;
       
        [time,YALL]=ode23s(@Press1, 0:delt:tpress_end,Y,options);
        t2press=cputime-t1;
        
        vin2=vpress;
        yin=(YFEED(1).*vin2*delz/2*Pe+YALL(:,1) )./ (1+vin2*delz/2*Pe);
        Tin=(1.0       .*vin2*delz/2.0*Peh+YALL(:,3*ngrid+1) )./ (1.0+vin2*delz/2.0*Peh);
        YALL(:,6*ngrid+1)=YALL(:,5*ngrid+1)+(1/GAMMA).*vpress.*delz/2.0;
        massinpress=Area*trapz(time*t0,YFEED(1)./Tin.*vin2*V0*BV.*YALL(:,6*ngrid+1)*PH./(RG.*T0));
        massinpressCO2tank=10*YFEED(1)-YALL(end,6*ngrid+6);
        
        tPRESS(lpress)=time(end);
        tpress=tPRESS(end);
        tpress_=tpress*t0;
        tplot=time;
        YALL1=YALL;
       
        
        %-----------------
        Y=YALL(end,1:6*ngrid);
        Y(6*ngrid+(1:2))=0;
        vinit=-GAMMA*2/delz*(YALL(end,5*ngrid+(1))-YALL(end,6*ngrid+1));vinit=0;
    
        options=odeset('RelTol', [1e-5], 'AbsTol', [1e-5], 'Vectorized','on','JPattern',jpattern2(ngrid));
    
        deltads=0.1;
        [time,YALL]=ode23s(@Ads2, 0:deltads:tads,Y,options);YALL2=YALL;
    
        vexit=-GAMMA*2.0/delz* (1-YALL(:,5*ngrid+ngrid)); vexit(1)=vinit;
        
        massinads=trapz(t0*time,YALL(:,5*ngrid+1).*(YFEED(1)/1-YALL(:,ngrid)*0))*PH/RG/T0*(1*vads*V0)*BV*Area;
    
        massoutads=trapz(t0*time,YALL(:,6*ngrid).*YALL(:,ngrid)./YALL(:,4*ngrid).*vexit)*PH/RG/T0*(1*vads*V0)*BV*Area;
        massoutadsCO2tank=YALL(end,6*ngrid+2);
    
        massoutads_total=trapz(t0*time,YALL(:,6*ngrid)./YALL(:,4*ngrid).*vexit)*PH/RG/T0*(1*vads*V0)*BV*Area;
        massoutads_totaltank=YALL(end,6*ngrid+1);
        
     %-----------------
     lblow=lblow+1;
       Y=YALL(end,1:6*ngrid);
      Y(6*ngrid+1)=PH/PH; %PBARBED initial
        Y(6*ngrid+(2:3))=0;
      vinit=-GAMMA*2.0/delz* (1-YALL(end,5*ngrid+ngrid));vinit=0;
       delt=0.01;
     options=odeset('Events',@events2,'RelTol', [1e-5], 'AbsTol', [1e-5], 'Vectorized','on','JPattern',jpattern3(ngrid));

       [time,YALL]=ode23s(@Blow3, 0:delt:tblow_end,Y,options);
      
       vexit2=vblow;
       YALL(:,6*ngrid+1)=YALL(:,5*ngrid+ngrid)-(1/GAMMA).*vblow.*delz/2.0;
   
        tBLOW(lblow)=time(end);
        tblow=tBLOW(end);
        tblow_=tblow*t0;
      
        massoutblow=trapz(time*t0, BV*vexit2*V0.*YALL(:,ngrid).*YALL(:,6*ngrid+1)*PH./YALL(:,4*ngrid)/T0 *Area/RG );
        massoutblowCO2tank=YALL(end,6*ngrid+3);
       massoutblow_totaltank=YALL(end,6*ngrid+2);
        
        YALL3=YALL;
%         massoutblowCO2tank = 0;
        %-----------------
        levac=levac+1;
        Y=YALL(end,1:6*ngrid);
        for i=6:-1:1
            abc((i-1)*ngrid+(1:ngrid))=Y(i*ngrid:-1:(i-1)*ngrid+1);
        end
        clear Y; Y=abc;clear abc;
        Y(6*ngrid+1)=PINT/PH; %PBARBED initial
        Y(6*ngrid+(2:3))=0;
        vinit=-GAMMA*2.0/delz* (1-YALL(end,5*ngrid+ngrid));vinit=0;
        delt=0.02;
        options=odeset('Events',@events3,'RelTol', [1e-5], 'AbsTol', [1e-5], 'Vectorized','on','JPattern',jpattern3(ngrid));
        [time,YALL]=ode23s(@Evac4, 0:delt:tevac_end,Y,options);
        
        vin=vevac;
        YALL(:,6*ngrid+1)=YALL(:,5*ngrid+ngrid)-(1/GAMMA).*vevac.*delz/2.0;
    
        tEVAC(levac)=time(end);
        tevac=tEVAC(end);
        tevac_=tevac*t0;
        
        massoutevac=trapz(time*t0, BV*vin*V0.*YALL(:,ngrid).*YALL(:,6*ngrid+1)*PH./YALL(:,4*ngrid)/T0 *Area/RG );
        massoutevacCO2tank=YALL(end,6*ngrid+3);
        massoutevac_totaltank=YALL(end,6*ngrid+2);
        
        
        puritylist(n)=(massoutevacCO2tank/massoutevac_totaltank)*100
        
        massin= massinpressCO2tank+massinads;
        massout=massoutadsCO2tank+massoutblowCO2tank+massoutevacCO2tank;
        
        recoverylist(n)=massoutevacCO2tank/(massinpressCO2tank+massinads)*100
        masslist(n)=abs((massin-massout)/massin)*100;
       
        for i=6:-1:1
            YALL(:,(i-1)*ngrid+(1:ngrid))=YALL(:,i*ngrid:-1:(i-1)*ngrid+1);
        end
        YALL4=YALL;
        %-----------------
        
         
        %% Cyclic steady state condition
        if n==1
           swtch=1;
        end
    
        if n>5 % will check after 50 cycles
           ind=find(masslist(end-5+1:end)<0.05);
           if length(ind)==5
              swtch=1;
           end
        end
end

purity=puritylist(end);
recovery=recoverylist(end);

save RST.mat

%----------
%FUNCTION
    function yprime=Press1(time,y)
        yprime=zeros(6*ngrid+6,size(y,2));
        
        y(6*ngrid+1,:)=y(5*ngrid+1,:)+(1/GAMMA).*vpress.*delz/2.0;
        PBARBED=y(6*ngrid+1,:);
        ve(ngrid,size(y,2))=0;
        delz=1/ngrid;
        ve0=vpress;
        ve0=ve0.*(ve0>0);
        
        
        AVGMW=y(1:ngrid,:)*MW(1)+(1-y(1:ngrid,:))*MW(2);
        ROWG=(PH.*y(5*ngrid+(1:ngrid),:).*AVGMW/1000)./(RG*T0.*y(3*ngrid+(1:ngrid),:)); %[kg/m^3] density of gas
        PII5=CPG/RG .*PH./T0 .*AVGMW/1000;
        PII7=GRP1*(-CPA)*QSref.*AVGMW/1000;
        
        %----under isothermads----
        BETA1(1:ngrid,:)=(y(5*ngrid+(1:ngrid),:)*PH)./(RG*y(3*ngrid+(1:ngrid),:)*T0).*(B0(1)*exp(-U1(1)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*1;
        BETA2(1:ngrid,:)=(y(5*ngrid+(1:ngrid),:)*PH)./(RG*y(3*ngrid+(1:ngrid),:)*T0).*(B0(2)*exp(-U1(2)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*1;
        DETA1(1:ngrid,:)=(y(5*ngrid+(1:ngrid),:)*PH)./(RG*y(3*ngrid+(1:ngrid),:)*T0).*(D0(1)*exp(-U2(1)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*1;
        DETA2(1:ngrid,:)=(y(5*ngrid+(1:ngrid),:)*PH)./(RG*y(3*ngrid+(1:ngrid),:)*T0).*(D0(2)*exp(-U2(2)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*1;
        
        DEN1(1:ngrid,:)=1+BETA1(1:ngrid,:).*y(1:ngrid,:)+BETA2.*(1-y(1:ngrid,:));
        DEN2(1:ngrid,:)=1+DETA1(1:ngrid,:).*y(1:ngrid,:)+DETA2.*(1-y(1:ngrid,:));
        xstar1(1:ngrid,:)=RATIO1(1).*BETA1(1:ngrid,:).*y(1:ngrid,:)./DEN1+RATIO2(1).*DETA1(1:ngrid,:).*y(1:ngrid,:)./DEN2;
        xstar2(1:ngrid,:)=RATIO1(2).*BETA2(1:ngrid,:).*(1-y(1:ngrid,:))./DEN1+RATIO2(2).*DETA2(1:ngrid,:).*(1-y(1:ngrid,:))./DEN2;
        %----
        QbyC1(1:ngrid,:)=(RATIO1(1).*B0(1)*exp(-U1(1)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*QSref./DEN1 +...
            (RATIO2(1).*D0(1)*exp(-U2(1)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*QSref./DEN2;
        QbyC2(1:ngrid,:)=(RATIO1(2).*B0(2)*exp(-U1(2)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*QSref./DEN1 +...
            (RATIO2(2).*D0(2)*exp(-U2(2)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*QSref./DEN2;
        ALFA1(1:ngrid,:)=15.0*PV*(DP*PREF./(y(5*ngrid+(1:ngrid),:)*PH))/(RP^2)./QbyC1 *BL/V0;
        ALFA2(1:ngrid,:)=15.0*PV*(DP*PREF./(y(5*ngrid+(1:ngrid),:)*PH))/(RP^2)./QbyC2 *BL/V0;

        
        yprime(1*ngrid+(1:ngrid),:)=ALFA1(1:ngrid,:).*(xstar1-y(1*ngrid+(1:ngrid),:));
        yprime(2*ngrid+(1:ngrid),:)=ALFA2(1:ngrid,:).*(xstar2-y(2*ngrid+(1:ngrid),:));
        
        
        %BC Y @0 and 1
        %component1 only
        yin=(YFEED(1).*ve0*delz/2*Pe+y(1,:) )./ (1+ve0*delz/2*Pe);
        
        %----
        sum5=yprime(ngrid+(1:ngrid),:)+yprime(2*ngrid+(1:ngrid),:);
        % sum4new=PI61*yprime(ngrid+(1:ngrid))'+PI62*yprime(2*ngrid+(1:ngrid))';
        %----Balance ads----
        
        
        Pin=PBARBED;
        P0=(2.0*Pin-y(5*ngrid+1,:));
        
        ryplus(1,:)=(y(5*ngrid+1,:)-P0(1,:)+10e-10)./(y(5*ngrid+2,:)-y(5*ngrid+1,:)+10e-10);
        ryplus(2:ngrid-1,:)=(y(5*ngrid+(2:ngrid-1),:)-y(5*ngrid+(1:ngrid-2),:)+10e-10)./(y(5*ngrid+(3:ngrid),:)-y(5*ngrid+(2:ngrid-1),:)+10e-10);
        ryplus(ngrid,:)=(y(5*ngrid+ngrid,:)-y(5*ngrid+ngrid-1,:)+10e-10)./(2.0*(1.0-y(5*ngrid+ngrid,:))+10e-10);
        limy=(ryplus+abs(ryplus))./(1+abs(ryplus)); %vanleer
        
        Fye(5*ngrid+(1:ngrid-1),:)=0.5.*limy(1:ngrid-1,:).*(y(5*ngrid+(2:ngrid),:)-y(5*ngrid+(1:ngrid-1),:));
        PE(1:ngrid-1,:)=y(5*ngrid+(1:ngrid-1),:)+Fye(5*ngrid+(1:ngrid-1),:);PE(ngrid,:)=y(5*ngrid+ngrid,:);
        
        
        %HR for y1
        y0(1,:)=(2*yin-y(1,:));
        ryplus(1,:)=(y(1,:)-y0+10e-10)./(y(2,:)-y(1,:)+10e-10);
        ryplus(2:ngrid-1,:)=(y(2:ngrid-1,:)-y(1:ngrid-2,:)+10e-10)./(y(3:ngrid,:)-y(2:ngrid-1,:)+10e-10);
        limy=(ryplus+abs(ryplus))./(1+abs(ryplus)); %vanleer
        Fye(1:ngrid-1,:)=0.5.*limy(1:ngrid-1,:).*(y(2:ngrid,:)-y(1:ngrid-1,:));
        y1E=y(1:ngrid-1,:)+Fye(1:ngrid-1,:); y1E(ngrid,:)=y(ngrid,:);
        
        
        Tin   = (1.0       .*ve0*delz/2.0*Peh+y(3*ngrid+1,:) )./ (1.0+ve0*delz/2.0*Peh);
        T00(1,:)=(2*Tin-y(3*ngrid+1,:));
        ryplus(1,:)=(y(3*ngrid+1,:)-T00+10e-10)./(y(3*ngrid+2,:)-y(3*ngrid+1,:)+10e-10);
        ryplus(2:ngrid-1,:)=(y(3*ngrid+2:4*ngrid-1,:)-y(3*ngrid+1:4*ngrid-2,:)+10e-10)./(y(3*ngrid+3:4*ngrid,:)-y(3*ngrid+2:4*ngrid-1,:)+10e-10);
        limy=(ryplus+abs(ryplus))./(1+abs(ryplus)); %vanleer
        Fye(3*ngrid+1:4*ngrid-1,:)=0.5.*limy(1:ngrid-1,:).*(y(3*ngrid+2:4*ngrid,:)-y(3*ngrid+1:4*ngrid-1,:));
        TE=y(3*ngrid+(1:ngrid-1),:)+Fye(3*ngrid+1:4*ngrid-1,:); TE(ngrid,:)=y(4*ngrid,:);
        
        
        %CASE2
        ve(1:ngrid-1,:)=-GAMMA/delz*(y(5*ngrid+(2:ngrid),:)-y(5*ngrid+(1:ngrid-1),:));
        ve(ngrid,:)=-GAMMA*2/delz*(PE(ngrid,:)-y(5*ngrid+(ngrid),:));
        
        
        %fundamental for P
        yprime(5*ngrid+(2:ngrid),:)=-y(3*ngrid+(2:ngrid),:)./delz.*(PE(2:ngrid,:).*ve(2:ngrid,:)./TE(2:ngrid,:)-PE(1:ngrid-1,:).*ve(1:ngrid-1,:)./TE(1:ngrid-1,:))...
            +y(5*ngrid+(2:ngrid),:)./y(3*ngrid+(2:ngrid),:).*yprime(3*ngrid+(2:ngrid),:)-SHI*y(3*ngrid+(2:ngrid),:).*sum5(2:ngrid,:);
        yprime(5*ngrid+(1),:)=-y(3*ngrid+(1),:)/delz.*(PE(1,:).*ve(1,:)./TE(1,:)-Pin.*ve0./Tin)...
            +y(5*ngrid+(1),:)./y(3*ngrid+(1),:).*yprime(3*ngrid+(1),:)-SHI*y(3*ngrid+(1),:).*sum5(1,:);
        
        %Fundamental for Y
        %d2Y/dZ2
        sm1(1,:)=y(3*ngrid+(1),:)./y(5*ngrid+(1),:).*(1.0/delz^2).*...
            (   PE(1,:)./TE(1,:).*(y(2,:)-y(1,:)) -  2*Pin./Tin.*(y(1,:)-yin)    );
        sm1(2:ngrid-1,:)=y(3*ngrid+(2:ngrid-1),:)./y(5*ngrid+(2:ngrid-1),:).*(1.0/delz^2).*...
            (   PE(2:ngrid-1,:)./TE(2:ngrid-1,:).*(y(3:ngrid,:)-y(2:ngrid-1,:)) -  PE(1:ngrid-2,:)./TE(1:ngrid-2,:).*(y(2:ngrid-1,:)-y(1:ngrid-2,:))    );
        sm1(ngrid,:)=y(3*ngrid+(ngrid),:)./y(5*ngrid+(ngrid),:).*(1.0/delz^2).*...
            (   PE(ngrid,:)./TE(ngrid,:).*(0) -  PE(ngrid-1,:)./TE(ngrid-1,:).*(y(ngrid,:)-y(ngrid-1,:))    );
        
        %dY/dZ
        sm2all(1,:)=y(3*ngrid+(1),:)./y(5*ngrid+(1),:)/delz.*(PE(1,:).*ve(1,:).*y1E(1,:)./TE(1,:) - Pin.*ve0.*yin./Tin );
        sm2all(2:ngrid,:)=y(3*ngrid+(2:ngrid),:)./y(5*ngrid+(2:ngrid),:)/delz.*(PE(2:ngrid,:).*ve(2:ngrid,:).*y1E(2:ngrid,:)./TE(2:ngrid,:) - PE(1:ngrid-1,:).*ve(1:ngrid-1,:).*y1E(1:ngrid-1,:)./TE(1:ngrid-1,:) );
        
        yprime(1:ngrid,:) = sm1/Pe-sm2all...
            -SHI*y(3*ngrid+(1:ngrid),:)./y(5*ngrid+(1:ngrid),:).*yprime(ngrid+(1:ngrid),:)...
            +y(1:ngrid,:)./y(3*ngrid+(1:ngrid),:).*yprime(3*ngrid+(1:ngrid),:)...
            -y(1:ngrid,:)./y(5*ngrid+(1:ngrid),:).*yprime(5*ngrid+(1:ngrid),:);
        
        %Fundamental for T
        %d2T/dZ2
        sm4(1,:)=(1.0/delz^2)*(y(3*ngrid+2,:)-3*y(3*ngrid+1,:)+2*Tin);
        sm4(2:ngrid-1,:)=(1.0/delz^2)*(y(3*ngrid+(3:ngrid),:)-2*y(3*ngrid+(2:ngrid-1),:)+y(3*ngrid+(1:ngrid-2),:));
        sm4(ngrid,:)=(1.0/delz^2)*(y(3*ngrid+ngrid-1,:)-y(3*ngrid+ngrid,:));
        
        sm5all(2:ngrid,:)=PII5(2:ngrid,:)./delz.*(PE(2:ngrid,:).*ve(2:ngrid,:)-PE(1:ngrid-1,:).*ve(1:ngrid-1,:));
        sm5all(1,:)=PII5(1,:)/delz.*(PE(1,:).*ve(1,:)-Pin.*ve0);
        
        sm6(1,:)=(1.0/delz^2)*(y(4*ngrid+2,:)-3*y(4*ngrid+1,:)+2*1);
        sm6(2:ngrid-1,:)=(1.0/delz^2)*(y(4*ngrid+(3:ngrid),:)-2*y(4*ngrid+(2:ngrid-1),:)+y(4*ngrid+(1:ngrid-2),:));
        sm6(ngrid,:)=(1.0/delz^2)*(y(4*ngrid+ngrid-1,:)-y(4*ngrid+ngrid,:));
        
        GRP3=GRP1*(ROWS*CPS+CPA*QSref.*AVGMW/1000.* (y(ngrid+(1:ngrid),:)+y(2*ngrid+(1:ngrid),:)) );
        sum4=(PII61+y(3*ngrid+(1:ngrid),:).*PII7).*yprime(ngrid+(1:ngrid),:) + (PII62+y(3*ngrid+(1:ngrid),:).*PII7).*yprime(2*ngrid+(1:ngrid),:);
        
        yprime(3*ngrid+(1:ngrid),:)=1./GRP3.*(PII4*sm4-sm5all+sum4-PII8*(y(3*ngrid+(1:ngrid),:)-y(4*ngrid+(1:ngrid),:)) -PII5.*yprime(5*ngrid+(1:ngrid),:)).*0;
        yprime(4*ngrid+(1:ngrid),:)=(PI1*sm6+PI2*(y(3*ngrid+(1:ngrid),:)-y(4*ngrid+(1:ngrid),:))-PI3*(y(4*ngrid+(1:ngrid),:)-Tatm/T0)).*0;
        
        
        
        %----
        %====
        
        
        yprime(6*ngrid+2,:)=ve0;
        yprime(6*ngrid+3,:)=yin;
        yprime(6*ngrid+4,:)=Tin;
        
        yprime(6*ngrid+5,:)=-ve0.*0.25*pi*(2*Ri)^2*BL*BV.*PBARBED.*PH./(RG*Tin*T0);
        yprime(6*ngrid+6,:)=-ve0.*0.25*pi*(2*Ri)^2*BL*BV.*PBARBED.*PH./(RG*Tin*T0) *YFEED(1);
        
        
    end

    function yprime=Ads2(time,y)
        yprime=zeros(6*ngrid+2,size(y,2)); %ADS
        ve(ngrid,size(y,2))=0;
        delz=1/ngrid;
        ve0=1; %ADS 
       
        
        AVGMW=y(1:ngrid,:)*MW(1)+(1-y(1:ngrid,:))*MW(2);
        ROWG=(PH.*y(5*ngrid+(1:ngrid),:).*AVGMW/1000)./(RG*T0.*y(3*ngrid+(1:ngrid),:)); %[kg/m^3] density of gas
        PII5=CPG/RG .*PH./T0 .*AVGMW/1000;
        PII7=GRP1*(-CPA)*QSref.*AVGMW/1000;
        
        %----under isothermads----
        BETA1(1:ngrid,:)=(y(5*ngrid+(1:ngrid),:)*PH)./(RG*y(3*ngrid+(1:ngrid),:)*T0).*(B0(1)*exp(-U1(1)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*1;
        BETA2(1:ngrid,:)=(y(5*ngrid+(1:ngrid),:)*PH)./(RG*y(3*ngrid+(1:ngrid),:)*T0).*(B0(2)*exp(-U1(2)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*1;
        DETA1(1:ngrid,:)=(y(5*ngrid+(1:ngrid),:)*PH)./(RG*y(3*ngrid+(1:ngrid),:)*T0).*(D0(1)*exp(-U2(1)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*1;
        DETA2(1:ngrid,:)=(y(5*ngrid+(1:ngrid),:)*PH)./(RG*y(3*ngrid+(1:ngrid),:)*T0).*(D0(2)*exp(-U2(2)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*1;
        
        DEN1(1:ngrid,:)=1+BETA1(1:ngrid,:).*y(1:ngrid,:)+BETA2.*(1-y(1:ngrid,:));
        DEN2(1:ngrid,:)=1+DETA1(1:ngrid,:).*y(1:ngrid,:)+DETA2.*(1-y(1:ngrid,:));
        xstar1(1:ngrid,:)=RATIO1(1).*BETA1(1:ngrid,:).*y(1:ngrid,:)./DEN1+RATIO2(1).*DETA1(1:ngrid,:).*y(1:ngrid,:)./DEN2;
        xstar2(1:ngrid,:)=RATIO1(2).*BETA2(1:ngrid,:).*(1-y(1:ngrid,:))./DEN1+RATIO2(2).*DETA2(1:ngrid,:).*(1-y(1:ngrid,:))./DEN2;
        %----
        QbyC1(1:ngrid,:)=(RATIO1(1).*B0(1)*exp(-U1(1)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*QSref./DEN1 +...
            (RATIO2(1).*D0(1)*exp(-U2(1)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*QSref./DEN2;
        QbyC2(1:ngrid,:)=(RATIO1(2).*B0(2)*exp(-U1(2)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*QSref./DEN1 +...
            (RATIO2(2).*D0(2)*exp(-U2(2)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*QSref./DEN2;
        ALFA1(1:ngrid,:)=15.0*PV*(DP*PREF./(y(5*ngrid+(1:ngrid),:)*PH))/(RP^2)./QbyC1 *BL/V0;
        ALFA2(1:ngrid,:)=15.0*PV*(DP*PREF./(y(5*ngrid+(1:ngrid),:)*PH))/(RP^2)./QbyC2 *BL/V0;

%         
        yprime(1*ngrid+(1:ngrid),:)=ALFA1(1:ngrid,:).*(xstar1-y(1*ngrid+(1:ngrid),:));
        yprime(2*ngrid+(1:ngrid),:)=ALFA2(1:ngrid,:).*(xstar2-y(2*ngrid+(1:ngrid),:));
        
        
        %BC Y @0 and 1
        %component1 only
        yin=(YFEED(1).*ve0*delz/2*Pe+y(1,:) )./ (1+ve0*delz/2*Pe);
        
        %----
        sum5=yprime(ngrid+(1:ngrid),:)+yprime(2*ngrid+(1:ngrid),:);
        %----Balance ads----
        
        
        
        Pin=y(5*ngrid+1,:)+ve0.*delz/2.0/GAMMA;   %ADS
        P0=(2.0*Pin-y(5*ngrid+1,:));
        
        ryplus(1,:)=(y(5*ngrid+1,:)-P0(1,:)+10e-10)./(y(5*ngrid+2,:)-y(5*ngrid+1,:)+10e-10);
        ryplus(2:ngrid-1,:)=(y(5*ngrid+(2:ngrid-1),:)-y(5*ngrid+(1:ngrid-2),:)+10e-10)./(y(5*ngrid+(3:ngrid),:)-y(5*ngrid+(2:ngrid-1),:)+10e-10);
        ryplus(ngrid,:)=(y(5*ngrid+ngrid,:)-y(5*ngrid+ngrid-1,:)+10e-10)./(2.0*(1.0-y(5*ngrid+ngrid,:))+10e-10); %ADS
        limy=(ryplus+abs(ryplus))./(1+abs(ryplus)); 
        
        Fye(5*ngrid+(1:ngrid-1),:)=0.5.*limy(1:ngrid-1,:).*(y(5*ngrid+(2:ngrid),:)-y(5*ngrid+(1:ngrid-1),:));
        Fye(5*ngrid+ngrid,:)=1.*(1.0-y(5*ngrid+ngrid,:)); %ADS
        PE(1:ngrid,:)=y(5*ngrid+(1:ngrid),:)+Fye(5*ngrid+(1:ngrid),:); %ADS
        
        
        %HR for y1
        y0(1,:)=(2*yin-y(1,:));
        ryplus(1,:)=(y(1,:)-y0+10e-10)./(y(2,:)-y(1,:)+10e-10);
        ryplus(2:ngrid-1,:)=(y(2:ngrid-1,:)-y(1:ngrid-2,:)+10e-10)./(y(3:ngrid,:)-y(2:ngrid-1,:)+10e-10);
        limy=(ryplus+abs(ryplus))./(1+abs(ryplus)); 
        Fye(1:ngrid-1,:)=0.5.*limy(1:ngrid-1,:).*(y(2:ngrid,:)-y(1:ngrid-1,:));
        y1E=y(1:ngrid-1,:)+Fye(1:ngrid-1,:); y1E(ngrid,:)=y(ngrid,:);
        
        
        Tin   = (1.0       .*ve0*delz/2.0*Peh+y(3*ngrid+1,:) )./ (1.0+ve0*delz/2.0*Peh);
        T00(1,:)=(2*Tin-y(3*ngrid+1,:));
        ryplus(1,:)=(y(3*ngrid+1,:)-T00+10e-10)./(y(3*ngrid+2,:)-y(3*ngrid+1,:)+10e-10);
        ryplus(2:ngrid-1,:)=(y(3*ngrid+2:4*ngrid-1,:)-y(3*ngrid+1:4*ngrid-2,:)+10e-10)./(y(3*ngrid+3:4*ngrid,:)-y(3*ngrid+2:4*ngrid-1,:)+10e-10);
        limy=(ryplus+abs(ryplus))./(1+abs(ryplus)); %vanleer
        Fye(3*ngrid+1:4*ngrid-1,:)=0.5.*limy(1:ngrid-1,:).*(y(3*ngrid+2:4*ngrid,:)-y(3*ngrid+1:4*ngrid-1,:));
        TE=y(3*ngrid+(1:ngrid-1),:)+Fye(3*ngrid+1:4*ngrid-1,:); TE(ngrid,:)=y(4*ngrid,:);
        
        
        
        
        %CASE2
        ve(1:ngrid-1,:)=-GAMMA/delz*(y(5*ngrid+(2:ngrid),:)-y(5*ngrid+(1:ngrid-1),:));
        ve(ngrid,:)=-GAMMA*2/delz*(PE(ngrid,:)-y(5*ngrid+(ngrid),:));
        
        
        %fundamental for P
        yprime(5*ngrid+(2:ngrid),:)=-y(3*ngrid+(2:ngrid),:)./delz.*(PE(2:ngrid,:).*ve(2:ngrid,:)./TE(2:ngrid,:)-PE(1:ngrid-1,:).*ve(1:ngrid-1,:)./TE(1:ngrid-1,:))...
            +y(5*ngrid+(2:ngrid),:)./y(3*ngrid+(2:ngrid),:).*yprime(3*ngrid+(2:ngrid),:)-SHI*y(3*ngrid+(2:ngrid),:).*sum5(2:ngrid,:);
        yprime(5*ngrid+(1),:)=-y(3*ngrid+(1),:)/delz.*(PE(1,:).*ve(1,:)./TE(1,:)-Pin.*ve0./Tin)...
            +y(5*ngrid+(1),:)./y(3*ngrid+(1),:).*yprime(3*ngrid+(1),:)-SHI*y(3*ngrid+(1),:).*sum5(1,:);
        
        %Fundamental for Y
        %d2Y/dZ2
        sm1(1,:)=y(3*ngrid+(1),:)./y(5*ngrid+(1),:).*(1.0/delz^2).*...
            (   PE(1,:)./TE(1,:).*(y(2,:)-y(1,:)) -  2*Pin./Tin.*(y(1,:)-yin)    );
        sm1(2:ngrid-1,:)=y(3*ngrid+(2:ngrid-1),:)./y(5*ngrid+(2:ngrid-1),:).*(1.0/delz^2).*...
            (   PE(2:ngrid-1,:)./TE(2:ngrid-1,:).*(y(3:ngrid,:)-y(2:ngrid-1,:)) -  PE(1:ngrid-2,:)./TE(1:ngrid-2,:).*(y(2:ngrid-1,:)-y(1:ngrid-2,:))    );
        sm1(ngrid,:)=y(3*ngrid+(ngrid),:)./y(5*ngrid+(ngrid),:).*(1.0/delz^2).*...
            (   PE(ngrid,:)./TE(ngrid,:).*(0) -  PE(ngrid-1,:)./TE(ngrid-1,:).*(y(ngrid,:)-y(ngrid-1,:))    );
        
        %dY/dZ
        sm2all(1,:)=y(3*ngrid+(1),:)./y(5*ngrid+(1),:)/delz.*(PE(1,:).*ve(1,:).*y1E(1,:)./TE(1,:) - Pin.*ve0.*yin./Tin );
        sm2all(2:ngrid,:)=y(3*ngrid+(2:ngrid),:)./y(5*ngrid+(2:ngrid),:)/delz.*(PE(2:ngrid,:).*ve(2:ngrid,:).*y1E(2:ngrid,:)./TE(2:ngrid,:) - PE(1:ngrid-1,:).*ve(1:ngrid-1,:).*y1E(1:ngrid-1,:)./TE(1:ngrid-1,:) );
        
        yprime(1:ngrid,:) = sm1/Pe-sm2all...
            -SHI*y(3*ngrid+(1:ngrid),:)./y(5*ngrid+(1:ngrid),:).*yprime(ngrid+(1:ngrid),:)...
            +y(1:ngrid,:)./y(3*ngrid+(1:ngrid),:).*yprime(3*ngrid+(1:ngrid),:)...
            -y(1:ngrid,:)./y(5*ngrid+(1:ngrid),:).*yprime(5*ngrid+(1:ngrid),:);
        
        %Fundamental for T
        %d2T/dZ2
        sm4(1,:)=(1.0/delz^2)*(y(3*ngrid+2,:)-3*y(3*ngrid+1,:)+2*Tin);
        sm4(2:ngrid-1,:)=(1.0/delz^2)*(y(3*ngrid+(3:ngrid),:)-2*y(3*ngrid+(2:ngrid-1),:)+y(3*ngrid+(1:ngrid-2),:));
        sm4(ngrid,:)=(1.0/delz^2)*(y(3*ngrid+ngrid-1,:)-y(3*ngrid+ngrid,:));
        
        sm5all(2:ngrid,:)=PII5(2:ngrid,:)./delz.*(PE(2:ngrid,:).*ve(2:ngrid,:)-PE(1:ngrid-1,:).*ve(1:ngrid-1,:));
        sm5all(1,:)=PII5(1,:)/delz.*(PE(1,:).*ve(1,:)-Pin.*ve0);
        
        sm6(1,:)=(1.0/delz^2)*(y(4*ngrid+2,:)-3*y(4*ngrid+1,:)+2*1);
        sm6(2:ngrid-1,:)=(1.0/delz^2)*(y(4*ngrid+(3:ngrid),:)-2*y(4*ngrid+(2:ngrid-1),:)+y(4*ngrid+(1:ngrid-2),:));
        sm6(ngrid,:)=(1.0/delz^2)*(y(4*ngrid+ngrid-1,:)-y(4*ngrid+ngrid,:));
        
        GRP3=GRP1*(ROWS*CPS+CPA*QSref.*AVGMW/1000.* (y(ngrid+(1:ngrid),:)+y(2*ngrid+(1:ngrid),:)) );
        sum4=(PII61+y(3*ngrid+(1:ngrid),:).*PII7).*yprime(ngrid+(1:ngrid),:) + (PII62+y(3*ngrid+(1:ngrid),:).*PII7).*yprime(2*ngrid+(1:ngrid),:);
        
        yprime(3*ngrid+(1:ngrid),:)=1./GRP3.*(PII4*sm4-sm5all+sum4-PII8*(y(3*ngrid+(1:ngrid),:)-y(4*ngrid+(1:ngrid),:)) -PII5.*yprime(5*ngrid+(1:ngrid),:)).*0;
        yprime(4*ngrid+(1:ngrid),:)=((PI1*sm6+PI2*(y(3*ngrid+(1:ngrid),:)-y(4*ngrid+(1:ngrid),:))-PI3*(y(4*ngrid+(1:ngrid),:)-Tatm/T0))).*0;
        
        yprime(6*ngrid+1,:)=ve(ngrid,:).*0.25*pi*(2*Ri)^2*BL*BV.*PE(ngrid,:).*PH./(RG*TE(ngrid,:)*T0);
        yprime(6*ngrid+2,:)=ve(ngrid,:).*0.25*pi*(2*Ri)^2*BL*BV.*PE(ngrid,:).*PH./(RG*TE(ngrid,:)*T0) *y1E(ngrid);
        
        
        
        %----
        %====
    end

    function yprime=Blow3(time,y)
        yprime=zeros(6*ngrid+3,size(y,2)); %BLOW
        ve(ngrid,size(y,2))=0;
        delz=1/ngrid;
        y(6*ngrid+1,:)=y(5*ngrid+ngrid,:)-(1/GAMMA).*vblow.*delz/2.0;
        PBARBED=y(6*ngrid+1,:);
        ve0=0; %BLOW
        
        AVGMW=y(1:ngrid,:)*MW(1)+(1-y(1:ngrid,:))*MW(2);
        ROWG=(PH.*y(5*ngrid+(1:ngrid),:).*AVGMW/1000)./(RG*T0.*y(3*ngrid+(1:ngrid),:)); %[kg/m^3] density of gas
        PII5=CPG/RG .*PH./T0 .*AVGMW/1000;
        PII7=GRP1*(-CPA)*QSref.*AVGMW/1000;
        
        %----under isothermads----
        BETA1(1:ngrid,:)=(y(5*ngrid+(1:ngrid),:)*PH)./(RG*y(3*ngrid+(1:ngrid),:)*T0).*(B0(1)*exp(-U1(1)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*1;
        BETA2(1:ngrid,:)=(y(5*ngrid+(1:ngrid),:)*PH)./(RG*y(3*ngrid+(1:ngrid),:)*T0).*(B0(2)*exp(-U1(2)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*1;
        DETA1(1:ngrid,:)=(y(5*ngrid+(1:ngrid),:)*PH)./(RG*y(3*ngrid+(1:ngrid),:)*T0).*(D0(1)*exp(-U2(1)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*1;
        DETA2(1:ngrid,:)=(y(5*ngrid+(1:ngrid),:)*PH)./(RG*y(3*ngrid+(1:ngrid),:)*T0).*(D0(2)*exp(-U2(2)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*1;
        
        DEN1(1:ngrid,:)=1+BETA1(1:ngrid,:).*y(1:ngrid,:)+BETA2.*(1-y(1:ngrid,:));
        DEN2(1:ngrid,:)=1+DETA1(1:ngrid,:).*y(1:ngrid,:)+DETA2.*(1-y(1:ngrid,:));
        xstar1(1:ngrid,:)=RATIO1(1).*BETA1(1:ngrid,:).*y(1:ngrid,:)./DEN1+RATIO2(1).*DETA1(1:ngrid,:).*y(1:ngrid,:)./DEN2;
        xstar2(1:ngrid,:)=RATIO1(2).*BETA2(1:ngrid,:).*(1-y(1:ngrid,:))./DEN1+RATIO2(2).*DETA2(1:ngrid,:).*(1-y(1:ngrid,:))./DEN2;
        %----
        QbyC1(1:ngrid,:)=(RATIO1(1).*B0(1)*exp(-U1(1)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*QSref./DEN1 +...
            (RATIO2(1).*D0(1)*exp(-U2(1)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*QSref./DEN2;
        QbyC2(1:ngrid,:)=(RATIO1(2).*B0(2)*exp(-U1(2)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*QSref./DEN1 +...
            (RATIO2(2).*D0(2)*exp(-U2(2)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*QSref./DEN2;
        ALFA1(1:ngrid,:)=15.0*PV*(DP*PREF./(y(5*ngrid+(1:ngrid),:)*PH))/(RP^2)./QbyC1 *BL/V0;
        ALFA2(1:ngrid,:)=15.0*PV*(DP*PREF./(y(5*ngrid+(1:ngrid),:)*PH))/(RP^2)./QbyC2 *BL/V0;

%         
        yprime(1*ngrid+(1:ngrid),:)=ALFA1(1:ngrid,:).*(xstar1-y(1*ngrid+(1:ngrid),:));
        yprime(2*ngrid+(1:ngrid),:)=ALFA2(1:ngrid,:).*(xstar2-y(2*ngrid+(1:ngrid),:));
        
        
        %BC Y @0 and 1
        %component1 only
        yin=(YFEED(1).*ve0*delz/2*Pe+y(1,:) )./ (1+ve0*delz/2*Pe);
        
        %----
        sum5=yprime(ngrid+(1:ngrid),:)+yprime(2*ngrid+(1:ngrid),:);
        %----Balance ads----
        
        
        
        Pin=y(5*ngrid+1,:)+ve0*delz/2.0/GAMMA;   %BLOW
        P0=(2.0*Pin-y(5*ngrid+1,:));
        
        ryplus(1,:)=(y(5*ngrid+1,:)-P0(1,:)+10e-10)./(y(5*ngrid+2,:)-y(5*ngrid+1,:)+10e-10);
        ryplus(2:ngrid-1,:)=(y(5*ngrid+(2:ngrid-1),:)-y(5*ngrid+(1:ngrid-2),:)+10e-10)./(y(5*ngrid+(3:ngrid),:)-y(5*ngrid+(2:ngrid-1),:)+10e-10);
        ryplus(ngrid,:)=(y(5*ngrid+ngrid,:)-y(5*ngrid+ngrid-1,:)+10e-10)./(2.0*(PBARBED-y(5*ngrid+ngrid,:))+10e-10); %BLOW
        limy=(ryplus+abs(ryplus))./(1+abs(ryplus)); %vanleer
        
        Fye(5*ngrid+(1:ngrid-1),:)=0.5.*limy(1:ngrid-1,:).*(y(5*ngrid+(2:ngrid),:)-y(5*ngrid+(1:ngrid-1),:));
        Fye(5*ngrid+ngrid,:)=1.*(PBARBED-y(5*ngrid+ngrid,:)); %BLOW
        PE(1:ngrid,:)=y(5*ngrid+(1:ngrid),:)+Fye(5*ngrid+(1:ngrid),:); %BLOW
        
        
        %HR for y1
        y0(1,:)=(2*yin-y(1,:));
        ryplus(1,:)=(y(1,:)-y0+10e-10)./(y(2,:)-y(1,:)+10e-10);
        ryplus(2:ngrid-1,:)=(y(2:ngrid-1,:)-y(1:ngrid-2,:)+10e-10)./(y(3:ngrid,:)-y(2:ngrid-1,:)+10e-10);
        limy=(ryplus+abs(ryplus))./(1+abs(ryplus)); %vanleer
        Fye(1:ngrid-1,:)=0.5.*limy(1:ngrid-1,:).*(y(2:ngrid,:)-y(1:ngrid-1,:));
        y1E=y(1:ngrid-1,:)+Fye(1:ngrid-1,:); y1E(ngrid,:)=y(ngrid,:);
        
        
        Tin   = (1.0       .*ve0*delz/2.0*Peh+y(3*ngrid+1,:) )./ (1.0+ve0*delz/2.0*Peh);
        T00(1,:)=(2*Tin-y(3*ngrid+1,:));
        ryplus(1,:)=(y(3*ngrid+1,:)-T00+10e-10)./(y(3*ngrid+2,:)-y(3*ngrid+1,:)+10e-10);
        ryplus(2:ngrid-1,:)=(y(3*ngrid+2:4*ngrid-1,:)-y(3*ngrid+1:4*ngrid-2,:)+10e-10)./(y(3*ngrid+3:4*ngrid,:)-y(3*ngrid+2:4*ngrid-1,:)+10e-10);
        limy=(ryplus+abs(ryplus))./(1+abs(ryplus)); %vanleer
        Fye(3*ngrid+1:4*ngrid-1,:)=0.5.*limy(1:ngrid-1,:).*(y(3*ngrid+2:4*ngrid,:)-y(3*ngrid+1:4*ngrid-1,:));
        TE=y(3*ngrid+(1:ngrid-1),:)+Fye(3*ngrid+1:4*ngrid-1,:); TE(ngrid,:)=y(4*ngrid,:);
        
        
        
        
        %CASE2
        ve(1:ngrid-1,:)=-GAMMA/delz*(y(5*ngrid+(2:ngrid),:)-y(5*ngrid+(1:ngrid-1),:));
        ve(ngrid,:)=-GAMMA*2/delz*(PBARBED-y(5*ngrid+(ngrid),:));
        
        
        
        %fundamental for P
        yprime(5*ngrid+(2:ngrid),:)=-y(3*ngrid+(2:ngrid),:)./delz.*(PE(2:ngrid,:).*ve(2:ngrid,:)./TE(2:ngrid,:)-PE(1:ngrid-1,:).*ve(1:ngrid-1,:)./TE(1:ngrid-1,:))...
            +y(5*ngrid+(2:ngrid),:)./y(3*ngrid+(2:ngrid),:).*yprime(3*ngrid+(2:ngrid),:)-SHI*y(3*ngrid+(2:ngrid),:).*sum5(2:ngrid,:);
        yprime(5*ngrid+(1),:)=-y(3*ngrid+(1),:)/delz.*(PE(1,:).*ve(1,:)./TE(1,:)-Pin.*ve0./Tin)...
            +y(5*ngrid+(1),:)./y(3*ngrid+(1),:).*yprime(3*ngrid+(1),:)-SHI*y(3*ngrid+(1),:).*sum5(1,:);
        
        %Fundamental for Y
        %d2Y/dZ2
        sm1(1,:)=y(3*ngrid+(1),:)./y(5*ngrid+(1),:).*(1.0/delz^2).*...
            (   PE(1,:)./TE(1,:).*(y(2,:)-y(1,:)) -  2*Pin./Tin.*(y(1,:)-yin)    );
        sm1(2:ngrid-1,:)=y(3*ngrid+(2:ngrid-1),:)./y(5*ngrid+(2:ngrid-1),:).*(1.0/delz^2).*...
            (   PE(2:ngrid-1,:)./TE(2:ngrid-1,:).*(y(3:ngrid,:)-y(2:ngrid-1,:)) -  PE(1:ngrid-2,:)./TE(1:ngrid-2,:).*(y(2:ngrid-1,:)-y(1:ngrid-2,:))    );
        sm1(ngrid,:)=y(3*ngrid+(ngrid),:)./y(5*ngrid+(ngrid),:).*(1.0/delz^2).*...
            (   PE(ngrid,:)./TE(ngrid,:).*(0) -  PE(ngrid-1,:)./TE(ngrid-1,:).*(y(ngrid,:)-y(ngrid-1,:))    );
        
        %dY/dZ
        sm2all(1,:)=y(3*ngrid+(1),:)./y(5*ngrid+(1),:)/delz.*(PE(1,:).*ve(1,:).*y1E(1,:)./TE(1,:) - Pin.*ve0.*yin./Tin );
        sm2all(2:ngrid,:)=y(3*ngrid+(2:ngrid),:)./y(5*ngrid+(2:ngrid),:)/delz.*(PE(2:ngrid,:).*ve(2:ngrid,:).*y1E(2:ngrid,:)./TE(2:ngrid,:) - PE(1:ngrid-1,:).*ve(1:ngrid-1,:).*y1E(1:ngrid-1,:)./TE(1:ngrid-1,:) );
        
        yprime(1:ngrid,:) = sm1/Pe-sm2all...
            -SHI*y(3*ngrid+(1:ngrid),:)./y(5*ngrid+(1:ngrid),:).*yprime(ngrid+(1:ngrid),:)...
            +y(1:ngrid,:)./y(3*ngrid+(1:ngrid),:).*yprime(3*ngrid+(1:ngrid),:)...
            -y(1:ngrid,:)./y(5*ngrid+(1:ngrid),:).*yprime(5*ngrid+(1:ngrid),:);
        
        %Fundamental for T
        %d2T/dZ2
        sm4(1,:)=(1.0/delz^2)*(y(3*ngrid+2,:)-3*y(3*ngrid+1,:)+2*Tin);
        sm4(2:ngrid-1,:)=(1.0/delz^2)*(y(3*ngrid+(3:ngrid),:)-2*y(3*ngrid+(2:ngrid-1),:)+y(3*ngrid+(1:ngrid-2),:));
        sm4(ngrid,:)=(1.0/delz^2)*(y(3*ngrid+ngrid-1,:)-y(3*ngrid+ngrid,:));
        
        sm5all(2:ngrid,:)=PII5(2:ngrid,:)./delz.*(PE(2:ngrid,:).*ve(2:ngrid,:)-PE(1:ngrid-1,:).*ve(1:ngrid-1,:));
        sm5all(1,:)=PII5(1,:)/delz.*(PE(1,:).*ve(1,:)-Pin.*ve0);
        
        sm6(1,:)=(1.0/delz^2)*(y(4*ngrid+2,:)-3*y(4*ngrid+1,:)+2*1);
        sm6(2:ngrid-1,:)=(1.0/delz^2)*(y(4*ngrid+(3:ngrid),:)-2*y(4*ngrid+(2:ngrid-1),:)+y(4*ngrid+(1:ngrid-2),:));
        sm6(ngrid,:)=(1.0/delz^2)*(y(4*ngrid+ngrid-1,:)-y(4*ngrid+ngrid,:));
        
        GRP3=GRP1*(ROWS*CPS+CPA*QSref.*AVGMW/1000.* (y(ngrid+(1:ngrid),:)+y(2*ngrid+(1:ngrid),:)) );
        sum4=(PII61+y(3*ngrid+(1:ngrid),:).*PII7).*yprime(ngrid+(1:ngrid),:) + (PII62+y(3*ngrid+(1:ngrid),:).*PII7).*yprime(2*ngrid+(1:ngrid),:);
        
        yprime(3*ngrid+(1:ngrid),:)=1./GRP3.*(PII4*sm4-sm5all+sum4-PII8*(y(3*ngrid+(1:ngrid),:)-y(4*ngrid+(1:ngrid),:)) -PII5.*yprime(5*ngrid+(1:ngrid),:)).*0;
        yprime(4*ngrid+(1:ngrid),:)=(PI1*sm6+PI2*(y(3*ngrid+(1:ngrid),:)-y(4*ngrid+(1:ngrid),:))-PI3*(y(4*ngrid+(1:ngrid),:)-Tatm/T0)).*0;
        
        yprime(6*ngrid+2,:)=ve(ngrid,:).*0.25*pi*(2*Ri)^2*BL*BV.*PH.*PBARBED./(RG*TE(ngrid,:)*T0);
        yprime(6*ngrid+3,:)=ve(ngrid,:).*0.25*pi*(2*Ri)^2*BL*BV.*PH.*PBARBED./(RG*TE(ngrid,:)*T0) *y1E(ngrid);
        
        
        %----
        %====
    end

    function yprime=Evac4(time,y)
        yprime=zeros(6*ngrid+3,size(y,2)); %BLOW
        ve(ngrid,size(y,2))=0;
        delz=1/ngrid;
        y(6*ngrid+1,:)=y(5*ngrid+ngrid,:)-(1/GAMMA).*vevac.*delz/2.0;
        PBARBED=y(6*ngrid+1,:);
        ve0=0; %BLOW
        
        AVGMW=y(1:ngrid,:)*MW(1)+(1-y(1:ngrid,:))*MW(2);
        ROWG=(PH.*y(5*ngrid+(1:ngrid),:).*AVGMW/1000)./(RG*T0.*y(3*ngrid+(1:ngrid),:)); %[kg/m^3] density of gas
        PII5=CPG/RG .*PH./T0 .*AVGMW/1000;
        PII7=GRP1*(-CPA)*QSref.*AVGMW/1000;
        
        %----under isothermads----
        BETA1(1:ngrid,:)=(y(5*ngrid+(1:ngrid),:)*PH)./(RG*y(3*ngrid+(1:ngrid),:)*T0).*(B0(1)*exp(-U1(1)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*1;
        BETA2(1:ngrid,:)=(y(5*ngrid+(1:ngrid),:)*PH)./(RG*y(3*ngrid+(1:ngrid),:)*T0).*(B0(2)*exp(-U1(2)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*1;
        DETA1(1:ngrid,:)=(y(5*ngrid+(1:ngrid),:)*PH)./(RG*y(3*ngrid+(1:ngrid),:)*T0).*(D0(1)*exp(-U2(1)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*1;
        DETA2(1:ngrid,:)=(y(5*ngrid+(1:ngrid),:)*PH)./(RG*y(3*ngrid+(1:ngrid),:)*T0).*(D0(2)*exp(-U2(2)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*1;
        
        DEN1(1:ngrid,:)=1+BETA1(1:ngrid,:).*y(1:ngrid,:)+BETA2.*(1-y(1:ngrid,:));
        DEN2(1:ngrid,:)=1+DETA1(1:ngrid,:).*y(1:ngrid,:)+DETA2.*(1-y(1:ngrid,:));
        xstar1(1:ngrid,:)=RATIO1(1).*BETA1(1:ngrid,:).*y(1:ngrid,:)./DEN1+RATIO2(1).*DETA1(1:ngrid,:).*y(1:ngrid,:)./DEN2;
        xstar2(1:ngrid,:)=RATIO1(2).*BETA2(1:ngrid,:).*(1-y(1:ngrid,:))./DEN1+RATIO2(2).*DETA2(1:ngrid,:).*(1-y(1:ngrid,:))./DEN2;
        %----
        QbyC1(1:ngrid,:)=(RATIO1(1).*B0(1)*exp(-U1(1)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*QSref./DEN1 +...
            (RATIO2(1).*D0(1)*exp(-U2(1)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*QSref./DEN2;
        QbyC2(1:ngrid,:)=(RATIO1(2).*B0(2)*exp(-U1(2)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*QSref./DEN1 +...
            (RATIO2(2).*D0(2)*exp(-U2(2)./(RG*y(3*ngrid+(1:ngrid),:)*T0)))*QSref./DEN2;
        ALFA1(1:ngrid,:)=15.0*PV*(DP*PREF./(y(5*ngrid+(1:ngrid),:)*PH))/(RP^2)./QbyC1 *BL/V0;
        ALFA2(1:ngrid,:)=15.0*PV*(DP*PREF./(y(5*ngrid+(1:ngrid),:)*PH))/(RP^2)./QbyC2 *BL/V0;

%         
        yprime(1*ngrid+(1:ngrid),:)=ALFA1(1:ngrid,:).*(xstar1-y(1*ngrid+(1:ngrid),:));
        yprime(2*ngrid+(1:ngrid),:)=ALFA2(1:ngrid,:).*(xstar2-y(2*ngrid+(1:ngrid),:));
        
        
        %BC Y @0 and 1
        %component1 only
        yin=(YFEED(1).*ve0*delz/2*Pe+y(1,:) )./ (1+ve0*delz/2*Pe);
        
        %----
        sum5=yprime(ngrid+(1:ngrid),:)+yprime(2*ngrid+(1:ngrid),:);
        %----Balance ads----
        
        
        
        Pin=y(5*ngrid+1,:)+ve0*delz/2.0/GAMMA;   %BLOW
        P0=(2.0*Pin-y(5*ngrid+1,:));
        
        ryplus(1,:)=(y(5*ngrid+1,:)-P0(1,:)+10e-10)./(y(5*ngrid+2,:)-y(5*ngrid+1,:)+10e-10);
        ryplus(2:ngrid-1,:)=(y(5*ngrid+(2:ngrid-1),:)-y(5*ngrid+(1:ngrid-2),:)+10e-10)./(y(5*ngrid+(3:ngrid),:)-y(5*ngrid+(2:ngrid-1),:)+10e-10);
        ryplus(ngrid,:)=(y(5*ngrid+ngrid,:)-y(5*ngrid+ngrid-1,:)+10e-10)./(2.0*(PBARBED-y(5*ngrid+ngrid,:))+10e-10); %BLOW
        limy=(ryplus+abs(ryplus))./(1+abs(ryplus)); %vanleer
        
        Fye(5*ngrid+(1:ngrid-1),:)=0.5.*limy(1:ngrid-1,:).*(y(5*ngrid+(2:ngrid),:)-y(5*ngrid+(1:ngrid-1),:));
        Fye(5*ngrid+ngrid,:)=1.*(PBARBED-y(5*ngrid+ngrid,:)); %BLOW
        PE(1:ngrid,:)=y(5*ngrid+(1:ngrid),:)+Fye(5*ngrid+(1:ngrid),:); %BLOW
        
        
        %HR for y1
        y0(1,:)=(2*yin-y(1,:));
        ryplus(1,:)=(y(1,:)-y0+10e-10)./(y(2,:)-y(1,:)+10e-10);
        ryplus(2:ngrid-1,:)=(y(2:ngrid-1,:)-y(1:ngrid-2,:)+10e-10)./(y(3:ngrid,:)-y(2:ngrid-1,:)+10e-10);
        limy=(ryplus+abs(ryplus))./(1+abs(ryplus)); %vanleer
        Fye(1:ngrid-1,:)=0.5.*limy(1:ngrid-1,:).*(y(2:ngrid,:)-y(1:ngrid-1,:));
        y1E=y(1:ngrid-1,:)+Fye(1:ngrid-1,:); y1E(ngrid,:)=y(ngrid,:);
        
        
        Tin   = (1.0       .*ve0*delz/2.0*Peh+y(3*ngrid+1,:) )./ (1.0+ve0*delz/2.0*Peh);
        T00(1,:)=(2*Tin-y(3*ngrid+1,:));
        ryplus(1,:)=(y(3*ngrid+1,:)-T00+10e-10)./(y(3*ngrid+2,:)-y(3*ngrid+1,:)+10e-10);
        ryplus(2:ngrid-1,:)=(y(3*ngrid+2:4*ngrid-1,:)-y(3*ngrid+1:4*ngrid-2,:)+10e-10)./(y(3*ngrid+3:4*ngrid,:)-y(3*ngrid+2:4*ngrid-1,:)+10e-10);
        limy=(ryplus+abs(ryplus))./(1+abs(ryplus)); %vanleer
        Fye(3*ngrid+1:4*ngrid-1,:)=0.5.*limy(1:ngrid-1,:).*(y(3*ngrid+2:4*ngrid,:)-y(3*ngrid+1:4*ngrid-1,:));
        TE=y(3*ngrid+(1:ngrid-1),:)+Fye(3*ngrid+1:4*ngrid-1,:); TE(ngrid,:)=y(4*ngrid,:);
        
        
        
        
        %CASE2
        ve(1:ngrid-1,:)=-GAMMA/delz*(y(5*ngrid+(2:ngrid),:)-y(5*ngrid+(1:ngrid-1),:));
        ve(ngrid,:)=-GAMMA*2/delz*(PBARBED-y(5*ngrid+(ngrid),:));
        
        
        
        %fundamental for P
        yprime(5*ngrid+(2:ngrid),:)=-y(3*ngrid+(2:ngrid),:)./delz.*(PE(2:ngrid,:).*ve(2:ngrid,:)./TE(2:ngrid,:)-PE(1:ngrid-1,:).*ve(1:ngrid-1,:)./TE(1:ngrid-1,:))...
            +y(5*ngrid+(2:ngrid),:)./y(3*ngrid+(2:ngrid),:).*yprime(3*ngrid+(2:ngrid),:)-SHI*y(3*ngrid+(2:ngrid),:).*sum5(2:ngrid,:);
        yprime(5*ngrid+(1),:)=-y(3*ngrid+(1),:)/delz.*(PE(1,:).*ve(1,:)./TE(1,:)-Pin.*ve0./Tin)...
            +y(5*ngrid+(1),:)./y(3*ngrid+(1),:).*yprime(3*ngrid+(1),:)-SHI*y(3*ngrid+(1),:).*sum5(1,:);
        
        %Fundamental for Y
        %d2Y/dZ2
        sm1(1,:)=y(3*ngrid+(1),:)./y(5*ngrid+(1),:).*(1.0/delz^2).*...
            (   PE(1,:)./TE(1,:).*(y(2,:)-y(1,:)) -  2*Pin./Tin.*(y(1,:)-yin)    );
        sm1(2:ngrid-1,:)=y(3*ngrid+(2:ngrid-1),:)./y(5*ngrid+(2:ngrid-1),:).*(1.0/delz^2).*...
            (   PE(2:ngrid-1,:)./TE(2:ngrid-1,:).*(y(3:ngrid,:)-y(2:ngrid-1,:)) -  PE(1:ngrid-2,:)./TE(1:ngrid-2,:).*(y(2:ngrid-1,:)-y(1:ngrid-2,:))    );
        sm1(ngrid,:)=y(3*ngrid+(ngrid),:)./y(5*ngrid+(ngrid),:).*(1.0/delz^2).*...
            (   PE(ngrid,:)./TE(ngrid,:).*(0) -  PE(ngrid-1,:)./TE(ngrid-1,:).*(y(ngrid,:)-y(ngrid-1,:))    );
        
        %dY/dZ
        sm2all(1,:)=y(3*ngrid+(1),:)./y(5*ngrid+(1),:)/delz.*(PE(1,:).*ve(1,:).*y1E(1,:)./TE(1,:) - Pin.*ve0.*yin./Tin );
        sm2all(2:ngrid,:)=y(3*ngrid+(2:ngrid),:)./y(5*ngrid+(2:ngrid),:)/delz.*(PE(2:ngrid,:).*ve(2:ngrid,:).*y1E(2:ngrid,:)./TE(2:ngrid,:) - PE(1:ngrid-1,:).*ve(1:ngrid-1,:).*y1E(1:ngrid-1,:)./TE(1:ngrid-1,:) );
        
        yprime(1:ngrid,:) = sm1/Pe-sm2all...
            -SHI*y(3*ngrid+(1:ngrid),:)./y(5*ngrid+(1:ngrid),:).*yprime(ngrid+(1:ngrid),:)...
            +y(1:ngrid,:)./y(3*ngrid+(1:ngrid),:).*yprime(3*ngrid+(1:ngrid),:)...
            -y(1:ngrid,:)./y(5*ngrid+(1:ngrid),:).*yprime(5*ngrid+(1:ngrid),:);
        
        %Fundamental for T
        %d2T/dZ2
        sm4(1,:)=(1.0/delz^2)*(y(3*ngrid+2,:)-3*y(3*ngrid+1,:)+2*Tin);
        sm4(2:ngrid-1,:)=(1.0/delz^2)*(y(3*ngrid+(3:ngrid),:)-2*y(3*ngrid+(2:ngrid-1),:)+y(3*ngrid+(1:ngrid-2),:));
        sm4(ngrid,:)=(1.0/delz^2)*(y(3*ngrid+ngrid-1,:)-y(3*ngrid+ngrid,:));
        
        sm5all(2:ngrid,:)=PII5(2:ngrid,:)./delz.*(PE(2:ngrid,:).*ve(2:ngrid,:)-PE(1:ngrid-1,:).*ve(1:ngrid-1,:));
        sm5all(1,:)=PII5(1,:)/delz.*(PE(1,:).*ve(1,:)-Pin.*ve0);
        
        sm6(1,:)=(1.0/delz^2)*(y(4*ngrid+2,:)-3*y(4*ngrid+1,:)+2*1);
        sm6(2:ngrid-1,:)=(1.0/delz^2)*(y(4*ngrid+(3:ngrid),:)-2*y(4*ngrid+(2:ngrid-1),:)+y(4*ngrid+(1:ngrid-2),:));
        sm6(ngrid,:)=(1.0/delz^2)*(y(4*ngrid+ngrid-1,:)-y(4*ngrid+ngrid,:));
        
        GRP3=GRP1*(ROWS*CPS+CPA*QSref.*AVGMW/1000.* (y(ngrid+(1:ngrid),:)+y(2*ngrid+(1:ngrid),:)) );
        sum4=(PII61+y(3*ngrid+(1:ngrid),:).*PII7).*yprime(ngrid+(1:ngrid),:) + (PII62+y(3*ngrid+(1:ngrid),:).*PII7).*yprime(2*ngrid+(1:ngrid),:);
        
        yprime(3*ngrid+(1:ngrid),:)=1./GRP3.*(PII4*sm4-sm5all+sum4-PII8*(y(3*ngrid+(1:ngrid),:)-y(4*ngrid+(1:ngrid),:)) -PII5.*yprime(5*ngrid+(1:ngrid),:)).*0;
        yprime(4*ngrid+(1:ngrid),:)=(PI1*sm6+PI2*(y(3*ngrid+(1:ngrid),:)-y(4*ngrid+(1:ngrid),:))-PI3*(y(4*ngrid+(1:ngrid),:)-Tatm/T0)).*0;
        
        yprime(6*ngrid+2,:)=ve(ngrid,:).*0.25*pi*(2*Ri)^2*BL*BV.*PH.*PBARBED./(RG*TE(ngrid,:)*T0);
        yprime(6*ngrid+3,:)=ve(ngrid,:).*0.25*pi*(2*Ri)^2*BL*BV.*PH.*PBARBED./(RG*TE(ngrid,:)*T0) *y1E(ngrid);
        
        
        %----
        %====
    end

    function [value,isterminal,direction] = events(time,y)
        
        value=y(5*ngrid+1)-1;
        isterminal = 1;
        direction=1;
        
    end

    function [value,isterminal,direction] = events2(time,y)
        
        value=y(5*ngrid+ngrid)-PINT/PH;
        isterminal = 1;
        direction=-1;
        
    end

    function [value,isterminal,direction] = events3(time,y)
        
        value=y(5*ngrid+ngrid)-PL/PH;
        isterminal = 1;
        direction=-1;
        
    end

    function S = jpattern1(ngrid)
        B=ones(ngrid,4);
        B1 = spdiags(B,-2:1,ngrid,ngrid);
        S=repmat(B1,[6 6]); S(6*ngrid+6,6*ngrid+6)=0;
        single=zeros(ngrid,6); single([1 2],:)=1; single=repmat(single,[6 1]);single(end+(1:6),:)=1;
        S(6*ngrid+(1:6),:)=single';
        S(:,6*ngrid+(1:6))=single;
        S=sparse(S);
    end

    function S = jpattern2(ngrid)
        B=ones(ngrid,4);
        B1 = spdiags(B,-2:1,ngrid,ngrid);
        S=repmat(B1,[6 6]);S(6*ngrid+2,6*ngrid+2)=0;
        single=zeros(ngrid,2); single([ngrid-1 ngrid],:)=1; single=repmat(single,[6 1]);single(end+(1:2),:)=1;
        S(6*ngrid+(1:2),:)=single';
        S(:,6*ngrid+(1:2))=single;
        S=sparse(S);
    end

    function S = jpattern3(ngrid)
        B=ones(ngrid,4);
        B1 = spdiags(B,-2:1,ngrid,ngrid);
        S=repmat(B1,[6 6]); S(6*ngrid+3,6*ngrid+3)=0;
        single=zeros(ngrid,3); single([ngrid-1 ngrid],:)=1; single=repmat(single,[6 1]);single(end+(1:3),:)=1;
        S(6*ngrid+(1:3),:)=single';
        S(:,6*ngrid+(1:3))=single;
        S=sparse(S);
       
    end

  

toc
end
