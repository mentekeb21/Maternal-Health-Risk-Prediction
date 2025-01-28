PROC IMPORT OUT=WORK.MaternalHealth
    DATAFILE= "C:\Users\Mentesnot\OneDrive\Desktop\Data Science Metro\SAS project\Maternal Health Risk Data Set.csv" 
    DBMS=CSV 
    REPLACE;
    GETNAMES=YES;
    DATAROW=2; 
RUN;


/********************************************************************************************
Features:
Age: Age in years when a woman is pregnant.
SystolicBP: Upper value of Blood Pressure in mmHg, another significant attribute during pregnancy.
DiastolicBP: Lower value of Blood Pressure in mmHg, another significant attribute during pregnancy.
BodyTemp :measured in degrees Fahrenheit, important for assessing maternal health and potential pregnancy risks.
BS: Blood glucose levels is in terms of a molar concentration, mmol/L.
HeartRate: A normal resting heart rate in beats per minute.
Risk Level: Predicted Risk Intensity Level during pregnancy considering the previous attribute.

Business question

"How do maternal health factors, such as age, blood pressure, body temperature, blood sugar, and heart rate, influence the predicted risk level during pregnancy?"
/* Displaying the Dataset */

PROC PRINT DATA=MaternalHealth (OBS=10);
run;

proc contents data=WORK.MaternalHealth;
run;


/*Feature Engenering*/

data MaternalHealthCategorized1;
    set MaternalHealth;

    /* Define age categories */
    if Age < 20 then AgeGroup = 'Under 20';
    else if Age < 30 then AgeGroup = '20-29';
    else if Age < 40 then AgeGroup = '30-39';
    *else if Age < 50 then AgeGroup = '40-49';
    else AgeGroup = '40 and Over';

    /*Interaction Terms*/
    BP_Interaction = SystolicBP * DiastolicBP;
    Age_BP_Interaction = Age * SystolicBP;

    /* Create a binary outcome for RiskLevel */
    if RiskLevel in ('high risk', 'mid risk') then CombinedRisk = 'At Risk';
    else CombinedRisk = 'Not At Risk';

run;

proc print data=MaternalHealthCategorized1(obs=20);
run;


proc contents data= MaternalHealthCategorized1;
run;

ods html;  /* Open the HTML output destination */

proc means data=MATERNALHEALTHCATEGORIZED1 mean median std min max;
    var Age SystolicBP DiastolicBP BS BodyTemp HeartRate;
run;
ods html close;

proc describe  data= MATERNALHEALTHCATEGORIZED1;
run;

PROC UNIVARIATE DATA=MaternalHealth;
    VAR Age BS BodyTemp DiastolicBP SystolicBP HeartRate;
RUN;

proc freq data=WORK.MATERNALHEALTHCATEGORIZED1;
    tables  AgeGroup CombinedRisk / nocum;
run;

PROC FREQ DATA=MaternalHealth;
    TABLES RiskLevel / MISSING;
    TITLE "Frequency Distribution of RiskLevel";
RUN;
   
ods html;  /* Open the HTML output destination */
/* Frequency distribution for AgeGroup */
proc freq data=MaternalHealthCategorized1;
    tables AgeGroup / missing;
    title "Frequency Distribution of AgeGroup";
run;


/* Frequency distribution for CombinedRisk */
proc freq data=MaternalHealthCategorized1;
    tables CombinedRisk / missing;
    title "Frequency Distribution of CombinedRisk";
run;

/*visualization for univarate analysis*/


PROC SGPLOT DATA=WORK.MATERNALHEALTHCATEGORIZED1;
    HISTOGRAM Age / TRANSPARENCY=0.5;
    DENSITY Age / TYPE=NORMAL;
    TITLE "Histogram with Normal Curve for Age";
RUN;
PROC SGPLOT DATA=WORK.MATERNALHEALTHCATEGORIZED1;
    VBOX Age / CATEGORY=CombinedRisk;
    TITLE "Box Plot of Age by CombinedRisk";
RUN;

/* Histogram and Box Plot for SystolicBP */
PROC SGPLOT DATA=WORK.MATERNALHEALTHCATEGORIZED1;
    HISTOGRAM SystolicBP / TRANSPARENCY=0.5;
    DENSITY SystolicBP / TYPE=NORMAL;
    TITLE "Histogram with Normal Curve for SystolicBP";
RUN;

PROC SGPLOT DATA=WORK.MATERNALHEALTHCATEGORIZED1;
    VBOX SystolicBP / CATEGORY=CombinedRisk;
    TITLE "Box Plot of SystolicBP by CombinedRisk";
RUN;


PROC SGPLOT DATA=WORK.MATERNALHEALTHCATEGORIZED1;
    HISTOGRAM DiastolicBP / TRANSPARENCY=0.5;
    DENSITY DiastolicBP / TYPE=NORMAL;
    TITLE "Histogram with Normal Curve for DiastolicBP";
RUN;

PROC SGPLOT DATA=WORK.MATERNALHEALTHCATEGORIZED1;
    VBOX DiastolicBP / CATEGORY=CombinedRisk;
    TITLE "Box Plot of DiastolicBP by CombinedRisk";
RUN;

PROC SGPLOT DATA=WORK.MATERNALHEALTHCATEGORIZED1;
    HISTOGRAM BS / TRANSPARENCY=0.5;
    DENSITY BS / TYPE=NORMAL;
    TITLE "Histogram with Normal Curve for Blood suger";
RUN;
PROC SGPLOT DATA=WORK.MATERNALHEALTHCATEGORIZED1;
    VBOX BS / CATEGORY=CombinedRisk;
    TITLE "Box Plot of Blood suger by CombinedRisk";
RUN;

PROC SGPLOT DATA=WORK.MATERNALHEALTHCATEGORIZED1;
    HISTOGRAM BodyTemp / TRANSPARENCY=0.5;
    DENSITY BodyTemp / TYPE=NORMAL;
    TITLE "Histogram with Normal Curve for BodyTemp";
RUN;
PROC SGPLOT DATA=WORK.MATERNALHEALTHCATEGORIZED1;
    VBOX BodyTemp / CATEGORY=CombinedRisk;
    TITLE "Box Plot of BodyTemp by CombinedRisk";
RUN;

PROC SGPLOT DATA=WORK.MATERNALHEALTHCATEGORIZED1;
    HISTOGRAM HeartRate / TRANSPARENCY=0.5;
    DENSITY HeartRate / TYPE=NORMAL;
    TITLE "Histogram with Normal Curve for HeartRate";
RUN;
PROC SGPLOT DATA=WORK.MATERNALHEALTHCATEGORIZED1;
    VBOX HeartRate / CATEGORY=CombinedRisk;
    TITLE "Box Plot of HeartRate by CombinedRisk";
RUN;


/* Step 1: Calculate Correlation Matrix */
proc corr data=WORK.MaternalHealthCategorized1 nosimple;
    var Age SystolicBP DiastolicBP BS HeartRate BodyTemp;
    ods output PearsonCorr=CorrMatrix;
run;

/* Step 2: Heatmap of Correlation Matrix */
proc sgplot data=CorrMatrix;
    heatmapparm x=_NAME_ y=Variable colorresponse=Corr / colormodel=(blue white red)
        outline fillattrs=(transparency=0.4);
    gradlegend / title="Correlation Coefficient";
    title "Correlation Heatmap for Continuous Variables";
    xaxis label="Variables";
    yaxis label="Variables";
run;



/* Cross-tabulation of RiskLevel against AgeGroup */
proc freq data=WORK.MaternalHealthCategorized1;
    tables CombinedRisk*AgeGroup / chisq norow nocol nopercent;
    title "Cross-tabulation of CombinedRisk and AgeGroup";
run;


/* ------ diagnostic ------*/

/* Multicol*/

data MaternalHealthCategorized2;
    set MaternalHealthCategorized1;
    if CombinedRisk = 'At Risk' then CombinedRiskNum = 1;
    else if CombinedRisk = 'Not At' then CombinedRiskNum = 0;
run;

proc reg data=MaternalHealthCategorized2;
    model CombinedRiskNum = Age SystolicBP DiastolicBP BS BodyTemp HeartRate / vif;
run;
quit;



/*missing data*/

PROC MEANS DATA=MaternalHealth NMISS;
    VAR Age BS BodyTemp DiastolicBP SystolicBP HeartRate;
    
RUN;


/*outlier detection */

/* Step 1: Calculate Q1 and Q3 for each variable */
/* Step 1: Calculate Q1 and Q3 for SystolicBP */


/* Step 1: Calculate Q1 and Q3 for SystolicBP */
proc univariate data=MaternalHealthCategorized1 noprint;
    var SystolicBP;
    output out=iqr_SBP q1=Q1_SBP q3=Q3_SBP;
run;

data outliers_SBP;
    set MaternalHealthCategorized1;
    if _n_ = 1 then set iqr_SBP;

    IQR_SBP = Q3_SBP - Q1_SBP;

    /* Identify outliers for SystolicBP using 3x IQR threshold */
    if SystolicBP < (Q1_SBP - 3*IQR_SBP) or SystolicBP > (Q3_SBP + 3*IQR_SBP) then Outlier_SBP = 1;
    else Outlier_SBP = 0;
run;

/* Step 2: Calculate Q1 and Q3 for DiastolicBP */
proc univariate data=MaternalHealthCategorized1 noprint;
    var DiastolicBP;
    output out=iqr_DBP q1=Q1_DBP q3=Q3_DBP;
run;

data outliers_DBP;
    set MaternalHealthCategorized1;
    if _n_ = 1 then set iqr_DBP;

    IQR_DBP = Q3_DBP - Q1_DBP;

    /* Identify outliers for DiastolicBP using 3x IQR threshold */
    if DiastolicBP < (Q1_DBP - 3*IQR_DBP) or DiastolicBP > (Q3_DBP + 3*IQR_DBP) then Outlier_DBP = 1;
    else Outlier_DBP = 0;
run;

/* Step 3: Calculate Q1 and Q3 for BS */
proc univariate data=MaternalHealthCategorized1 noprint;
    var BS;
    output out=iqr_BS q1=Q1_BS q3=Q3_BS;
run;

data outliers_BS;
    set MaternalHealthCategorized1;
    if _n_ = 1 then set iqr_BS;

    IQR_BS = Q3_BS - Q1_BS;

    /* Identify outliers for BS using 3x IQR threshold */
    if BS < (Q1_BS - 3*IQR_BS) or BS > (Q3_BS + 3*IQR_BS) then Outlier_BS = 1;
    else Outlier_BS = 0;
run;

/* Step 4: Calculate Q1 and Q3 for BodyTemp */
proc univariate data=MaternalHealthCategorized1 noprint;
    var BodyTemp;
    output out=iqr_BT q1=Q1_BT q3=Q3_BT;
run;

data outliers_BT;
    set MaternalHealthCategorized1;
    if _n_ = 1 then set iqr_BT;

    IQR_BT = Q3_BT - Q1_BT;

    /* Identify outliers for BodyTemp using 3x IQR threshold */
    if BodyTemp < (Q1_BT - 3*IQR_BT) or BodyTemp > (Q3_BT + 3*IQR_BT) then Outlier_BT = 1;
    else Outlier_BT = 0;
run;

/* Step 5: Calculate Q1 and Q3 for HeartRate */
proc univariate data=MaternalHealthCategorized1 noprint;
    var HeartRate;
    output out=iqr_HR q1=Q1_HR q3=Q3_HR;
run;

data outliers_HR;
    set MaternalHealthCategorized1;
    if _n_ = 1 then set iqr_HR;

    IQR_HR = Q3_HR - Q1_HR;

    /* Identify outliers for HeartRate using 3x IQR threshold */
    if HeartRate < (Q1_HR - 3*IQR_HR) or HeartRate > (Q3_HR + 3*IQR_HR) then Outlier_HR = 1;
    else Outlier_HR = 0;
run;

/* Step 6: Calculate Q1 and Q3 for Age */
proc univariate data=MaternalHealthCategorized1 noprint;
    var Age;
    output out=iqr_Age q1=Q1_Age q3=Q3_Age;
run;

data outliers_Age;
    set MaternalHealthCategorized1;
    if _n_ = 1 then set iqr_Age;

    IQR_Age = Q3_Age - Q1_Age;

    /* Identify outliers for Age using 3x IQR threshold */
    if Age < (Q1_Age - 3*IQR_Age) or Age > (Q3_Age + 3*IQR_Age) then Outlier_Age = 1;
    else Outlier_Age = 0;
run;

/* Combine outliers datasets */
data combined_outliers;
    merge outliers_SBP outliers_DBP outliers_BS 
          outliers_BT outliers_HR outliers_Age;
run;

/* Print the combined outliers */
proc print data = combined_outliers(obs=10);
run;

proc print data= MaternalHealthCategorized1;
run;

data outliers_only;
    set combined_outliers;
    /* Include rows where any of the outlier flags is 1 */
    if Outlier_SBP = 1 or Outlier_DBP = 1 or Outlier_BS = 1 
       or Outlier_BT = 1 or Outlier_HR = 1 or Outlier_Age = 1;
run;

/* Print the outliers (you can change obs=10 to display more or fewer rows) */
proc print data=outliers_only();
run;







/* Step 8: Cleaned dataset excluding outliers */
data MaternalHealthCleaned;
    set combined_outliers;
    /* Exclude records where any of the outlier flags is 1 */
    if Outlier_SBP = 0 and Outlier_DBP = 0 and Outlier_BS = 0 
       and Outlier_BT = 0 and Outlier_HR = 0 and Outlier_Age = 0;
run;

proc print data = MaternalHealthCleaned(obs=10);
run;

data main_columns_only;
    set MaternalHealthCleaned; /* Replace original_dataset with the name of your dataset */
    keep Obs Age SystolicBP DiastolicBP BS BodyTemp HeartRate RiskLevel AgeGroup BP_Interaction Age_BP_Interaction CombinedRisk ; /* Specify the columns to keep */
run;  

proc print data=main_columns_only(obs=10);
run;

data outliers_only;
    set combined_outliers;
    /* Include rows where any of the outlier flags is 1 */
    if Outlier_SBP = 1 or Outlier_DBP = 1 or Outlier_BS = 1 
       or Outlier_BT = 1 or Outlier_HR = 1 or Outlier_Age = 1;
run;

/* Print the outliers (you can change obs=10 to display more or fewer rows) */
proc print data=outliers_only();
run;

data main_columns;
    set combined_outliers; 
    keep Obs Age SystolicBP DiastolicBP BS BodyTemp HeartRate RiskLevel AgeGroup BP_Interaction Age_BP_Interaction CombinedRisk ; /* Specify the columns to keep */
run;







/*Bivaret anlysis*/
/*bivarte analysis*/ 


/*Hypothesis:
Null Hypothesis 
Maternal health factors (age, blood pressure, body temperature, blood sugar, and heart rate)
have no significant effect on the predicted risk level during pregnancy.*/


/*Assumptions for parametric tests 
Normality: The data for each group should be normally distributed.
Homogeneity of Variances: The variance among groups should be equal.
Independence: Observations should be independent of each other.*/


/*Testing for Normality*/

proc univariate data=main_columns normal;
    var Age SystolicBP DiastolicBP BS BodyTemp HeartRate  BP_Interaction Age_BP_Interaction;
    title "Shapiro-Wilk Test for Normality";
run;




proc univariate data=main_columns;
    var Age SystolicBP DiastolicBP BS BodyTemp HeartRate  BP_Interaction Age_BP_Interaction;
    histogram / normal;
    qqplot / normal;
    title "Histograms and Q-Q Plots for Continuous Variables";
run;

/*Testing for Homogeneity of Variances*/

proc glm data=main_columns;
    class RiskLevel;
    model Age SystolicBP DiastolicBP BS BodyTemp HeartRate  BP_Interaction Age_BP_Interaction; = CombinedRisk;
    means RiskLevel / hovtest=levene;
    title "Levene's Test for Homogeneity of Variances";
run;

/*Summary:
No Variables that meet the assumption of homogeneity: 


/*Logarithmic Transformation*/


/*---------Transformation---------*/


data log_transformed;
    set main_columns;
    
    /* Log transformation */
    Log_SystolicBP = log(SystolicBP+1);
	Log_Age = log(Age+1);
    Log_DiastolicBP = log(DiastolicBP+1);
    Log_BS = log(BS+1);
    Log_BodyTemp = log(BodyTemp+1);
	Log_HeartRate = log(HeartRate+1);
	Log_BP_Interaction = log(BP_Interaction+1);
	Log_Age_BP_Interaction = log(Age_BP_Interaction+1);
run;


/* Check Normality using Shapiro-Wilk Test */
proc univariate data=log_transformed normal;
    var Log_SystolicBP Log_Age Log_DiastolicBP Log_BS Log_BodyTemp Log_HeartRate;
    qqplot / normal (mu=est sigma=est);
run;





data MaternalHealthTransformed;
    set main_columns;
    
    /* Square root transformation */
    Sqrt_SystolicBP = sqrt(SystolicBP);
    Sqrt_DiastolicBP = sqrt(DiastolicBP);
    Sqrt_BS = sqrt(BS);
    Sqrt_BodyTemp = sqrt(BodyTemp);
	Sqrt_BP_Interaction = sqrt(BP_Interaction+1);
	Sqrt_Age_BP_Interaction = sqrt(Age_BP_Interaction+1);
run;

run;
proc print data=MaternalHealthTransformed (obs=10);



/* Use PROC TRANSREG to perform Box-Cox transformation */
proc transreg data=MaternalHealthCategorized1;
    model boxcox(Age SystolicBP DiastolicBP BS BodyTemp HeartRate);
    output out=boxcox_transformed;
run;



/* Create a dummy variable to allow the transformation */
data MaternalHealthCategorized1;
    set main_columns;
    dummy = 1; /* A constant variable used as the independent variable */
run;

/* Use PROC TRANSREG to perform Box-Cox transformation on independent variables */
proc transreg data=main_columns;
    model boxcox(Age SystolicBP DiastolicBP BS BodyTemp HeartRate BP_Interaction Age_BP_Interaction) = identity(dummy);
    output out=boxcox_transformed;
run;


/* Check normality of transformed variables */
proc univariate data=boxcox_transformed normal;
    var Age SystolicBP DiastolicBP BS BodyTemp HeartRate BP_Interaction Age_BP_Interaction;
    histogram / normal;
    probplot / normal(mu=est sigma=est);
run;

/* still cant transorm the data so i better use non parametric test

/* Kruskal-Wallis Test instade of ANOVA*/
/* Kruskal-Wallis Test */

proc npar1way data=log_transformed wilcoxon;
    class RiskLevel; /* Assuming 'RiskLevel' is your categorical variable with multiple groups */
    var Log_SystolicBP Log_Age Log_DiastolicBP Log_BS Log_BodyTemp Log_HeartRate;
run;

/* Wilcoxon Rank-Sum Test /T-Test*/
/* Wilcoxon Rank-Sum Test */
proc npar1way data=log_transformed wilcoxon;
    class CombinedRisk; /* Assuming 'RiskLevel' has two groups for comparison */
    var Log_SystolicBP Log_Age Log_DiastolicBP Log_BS Log_BodyTemp Log_HeartRate;
run;


/* Spearman's Rank Correlation /Pearson correlation/*/
/* Spearman's Rank Correlation */
proc corr data=log_transformed spearman;
    var Log_SystolicBP Log_Age Log_DiastolicBP Log_BS Log_BodyTemp Log_HeartRate;
run;

/* Assuming your dataset is named 'your_dataset' */
proc sgplot data=log_transformed;
    scatter x=Log_SystolicBP y=Log_DiastolicBP / group=CombinedRisk 
        markerattrs=(symbol=CircleFilled size=10) 
        transparency=0.1;
    xaxis label="Systolic Blood Pressure (mmHg)";
    yaxis label="Diastolic Blood Pressure (mmHg)";
    title "Systolic BP vs Diastolic BP by Risk Level";
run;
/*Across both AgeGroup and CombinedRisk, there are strong statistical indications that variables like SystolicBP, DiastolicBP, Age, and others differ significantly.
The p-values are all < 0.0001, which means the differences are highly significant, suggesting that factors such as age and risk status impact these health metrics.
If you are preparing a summary or report, you could highlight these key statistical results and emphasize the significant health differences across both age and risk groups.*/







/*  Scale Continuous Variables */


/* First, create a dataset with only the minority class (At Risk) */
data minority_class;
    set log_transformed;
    if CombinedRisk = 'At Risk';
run;

/* Create a dataset with the majority class (Not At Risk) */
data majority_class;
    set log_transformed;
    if CombinedRisk = 'Not At';
run;

/* Determine the number of observations needed for the minority class to match the majority class */
proc sql noprint;
    select count(*) into :majority_count from majority_class;
quit;

/* Over-sample the minority class */
data balanced_data;
    set majority_class;
    output; /* Always include the majority class */
    /* Over-sampling the minority class */
    do i = 1 to &majority_count / (267); /* Adjust this if you want a different ratio */
        set minority_class; /* Over-sample the minority class */
        output;
    end;
run;

/* Check the balance of the new dataset */
proc freq data=balanced_data;
    tables CombinedRisk;
run;

proc print data = balanced_data(obs=5);
run;


data filtered_dataset;
    set balanced_data; 
    keep AgeGroup BP_Interaction Age_BP_Interaction CombinedRisk 
         Log_SystolicBP Log_Age Log_DiastolicBP Log_BS 
         Log_BodyTemp Log_HeartRate;
run;

proc print data = filtered_dataset(obs=5);
run;


data transformed_dataset;
    set filtered_dataset;
   
    /* Keep only relevant variables */
    keep AgeGroup BP_Interaction Age_BP_Interaction CombinedRisk Log_SystolicBP Log_Age Log_DiastolicBP Log_BS Log_BodyTemp Log_HeartRate;
run;

proc print data= transformed_dataset(obs=5);
run;

proc standard data=transformed_dataset mean=0 std=1 out=maternal_health_scaled;
    var BP_Interaction Age_BP_Interaction Log_SystolicBP Log_Age Log_DiastolicBP Log_BS Log_BodyTemp Log_HeartRate; /* Include all continuous variables */
run;


/*-------Modeling---------*/


ods html;
proc surveyselect data=maternal_health_scaled rate=0.70 outall out=result seed=12345; 
run;

data traindata testdata;
    set result;
    if selected=1 then output traindata;
    else output testdata;
run;



ods graphics on; 
proc logistic data=traindata plots=(ROC);
    class AgeGroup(ref="40 and O") / param=ref;
    model CombinedRisk(event="At Risk") = AgeGroup BP_Interaction Age_BP_Interaction Log_SystolicBP Log_Age Log_DiastolicBP Log_BS Log_HeartRate; 
    /*details lackfit;*/
    score data=testdata out=testpred outroc=vroc;
    roc; 
    roccontrast;
    output out=outputedata p=prob_predicted xbeta=linpred;
run; 
quit;
ods graphics off;



  








/*-----------------------------detailed ------------------*/

/* Step 1: Sort the test predictions */
proc sort data=testpred; /* Assuming testpred contains the actual vs predicted outcomes */
    by descending F_CombinedRisk descending I_CombinedRisk; /* Adjust to your actual variable names */
run;

/* Step 2: Create the confusion matrix */
ods html on style=journal;

/* Generate the confusion matrix */
proc freq data=testpred order=data;
    tables F_CombinedRisk*I_CombinedRisk / out=CellCounts; /* Adjust variable names as needed */
run;

/* Step 3: Add a match column */
data CellCounts;
    set CellCounts;
    Match = (F_CombinedRisk = I_CombinedRisk); /* 1 if match, 0 otherwise */
run;

/* Step 4: Calculate overall match rate */
proc means data=CellCounts mean;
    freq count;
    var Match;
run;

ods html close;

/* Step 5: Calculate sensitivity and specificity */
ods html on;
ods graphics on; 

proc logistic data=traindata plots=ROC; 
    class AgeGroup(ref="40 and O") / param=ref;
    model CombinedRisk(event="At Risk") = AgeGroup BP_Interaction Age_BP_Interaction Log_SystolicBP Log_Age Log_DiastolicBP Log_BS Log_HeartRate 
         / details lackfit outroc=troc;
    score data=testdata out=testpred outroc=vroc; /* Ensure testdata has actual outcomes */
    roc; 
    roccontrast;
    output out=outputedata p=prob_predicted xbeta=linpred; /* Adjust as necessary */
run; 
quit;

ods graphics off;
ods html close;

/* Step 6: Sort frequency data for sensitivity analysis */
ods html style=journal;
ods select all;

/* Sorting for frequency data */
proc sort data=freq; /* Ensure freq dataset is generated and has correct variable names */
    by descending I_CombinedRisk descending F_CombinedRisk;
run;

proc sort data=testpred; /* Again, ensure testpred has the correct structure */
    by descending I_CombinedRisk descending F_CombinedRisk;
run;

/* Generate sensitivity and specificity table */
proc freq data=testpred order=data;
    tables F_CombinedRisk*I_CombinedRisk / senspec; /* Ensure variable names are correct */
run;

ods html close;



ods html close;
proc glmselect data=traindata plots=all;
    model CombinedRisk(event='At Risk') = Age SystolicBP DiastolicBP BS BodyTemp HeartRate 
        AgeGroup TempCategory HeartRateCategory
        / selection=lasso(choose=CV stop=none)
          cvmethod=random(10)
          display=summary;
    output out=lasso_out p=predicted_prob;
run;


ods html;
ods graphics on;
proc hpgenselect data=traindata;
    class AgeGroup TempCategory HeartRateCategory; /* Update classes */
    model CombinedRisk(event="At Risk")= AgeGroup TempCategory HeartRateCategory SystolicBP DiastolicBP BS BodyTemp HeartRate / distribution=binary cl;
    selection method=LASSO(choose=SBC) details=all;
    output out=out1 p=predlasso;
run;
ods graphics on;
ods html close;


proc contents data=traindata; 
run;

proc print data=traindata (obs=10); /* Print the first 10 observations */
run;



/*------------------threshold adj-----------*/











data testdata_with_probs;
    set testpred; /* Assuming testpred contains your predicted probabilities */
    
    /* Adjust the threshold for classification */
    if prob_predicted >= 0.6 then I_CombinedRisk = 'At Risk'; /* Change the threshold as needed */
    else I_CombinedRisk = 'Not At';
run;


proc freq data=testdata_with_probs;
    tables CombinedRisk*I_CombinedRisk / chisq;
    title "Confusion Matrix for Adjusted Threshold";
run;

/* Calculate Sensitivity, Specificity, PPV, and NPV */
proc freq data=testdata_with_probs;
    tables CombinedRisk*I_CombinedRisk / out=confusion_matrix;
run;

data stats;
    set confusion_matrix;
    if CombinedRisk = 'At Risk' and I_CombinedRisk = 'At Risk' then TP + 1; /* True Positive */
    else if CombinedRisk = 'At Risk' and I_CombinedRisk = 'Not At' then FN + 1; /* False Negative */
    else if CombinedRisk = 'Not At' and I_CombinedRisk = 'At Risk' then FP + 1; /* False Positive */
    else if CombinedRisk = 'Not At' and I_CombinedRisk = 'Not At' then TN + 1; /* True Negative */
run;

data performance;
    set stats;
    Sensitivity = TP / (TP + FN);
    Specificity = TN / (TN + FP);
    PositivePredictiveValue = TP / (TP + FP);
    NegativePredictiveValue = TN / (TN + FN);
run;

proc print data=performance; 
    title "Model Performance Metrics with Adjusted Threshold";
run;

/*-------------------------------GAM ----------------------------------*/

/* Step 1: ODS HTML and Graphics ON */
ods html;
ods graphics on; 

/* Step 2: Fit a GAM model to the training data */
proc gam data=traindata plots=all; 
    class AgeGroup TempCategory HeartRateCategory; /* Adjust to your categorical variables */
    model CombinedRisk(event="At Risk") = 
        param(AgeGroup TempCategory HeartRateCategory SystolicBP DiastolicBP BS BodyTemp HeartRate) 
        spline(Age, df=2) /* Adjust to your continuous variables */
        / dist=binomial;
    score data=testdata out=outputedata; /* Score the test data */
    output out=outputedata_gam p=prob_predicted all; /* Store predictions */
run; 
quit;

/* Step 3: Turn off ODS graphics */
ods graphics off;

/* Step 4: ODS Graphics ON for another GAM model */
ods graphics on; 

proc gam data=traindata plots=all; 
    class AgeGroup TempCategory HeartRateCategory; /* Adjust to your categorical variables */
    
    /* Model with spline for age and other covariates */
    model CombinedRisk(event="At Risk") = 
        spline(Age, df=2) 
        param(AgeGroup TempCategory HeartRateCategory SystolicBP DiastolicBP BS BodyTemp HeartRate); 

    score data=testdata out=outputedata_gam2; /* Score the test data */
    output out=pred p=phat; /* Store predictions */
run; 

/* Step 5: Turn off ODS graphics */
ods graphics off;

/* Step 6: GAM ROC Analysis */
ods graphics on;

proc logistic data=outputedata2; /* Adjust to your correct output dataset */
    where phatsurvived ne .; /* Ensure predictions are valid */
    class AgeGroup TempCategory HeartRateCategory; /* Use same classes as before */
    
    baseline_model: model CombinedRisk(event="At Risk")=; /* Specify the model for baseline */
    roc 'AUC of the Hold Out Sample for GAM' pred=phatsurvived; /* ROC for GAM predictions */
run; 

/* Step 7: Turn off ODS graphics */
ods graphics off;
ods html close; /* Close the ODS HTML output */




/**************************** Decision Tree Model with HPSPLIT **********************************/
ods graphics on;

proc hpsplit data=maternal_health_scaled; 
    class CombinedRisk AgeGroup; /* Specify categorical variables */
    model CombinedRisk(event="At Risk") = AgeGroup BP_Interaction Age_BP_Interaction Log_SystolicBP Log_Age Log_DiastolicBP Log_BS Log_HeartRate;; 
    prune costcomplexity; /* Prune tree using cost-complexity criterion */
    code file="C:\Users\Mentesnot\OneDrive\Desktop\Data Science Metro\SAS project\treeoutput.sas"; /* Output tree code */
    output out=scored; /* Save scored dataset */
run; 
quit;

ods graphics off;

/************************************* Scoring with the Decision Tree ************************************/
data tree_test_score;
    set testdata; /* Scoring the test data */
    %include 'C:\Users\Mentesnot\OneDrive\Desktop\Data Science Metro\SAS project\treeoutput.sas'; /* Scoring logic from the decision tree */
run;



/* Enable ODS Graphics */
ods graphics on;

/* Run HPSPLIT Procedure and Capture Tree Plot */
proc HPSPLIT data=MaternalHealthCategorized1; 
    class CombinedRisk AgeGroup TempCategory HeartRateCategory; 
    model CombinedRisk(event="At Risk")= AgeGroup TempCategory HeartRateCategory SystolicBP DiastolicBP BS BodyTemp HeartRate;
    prune costcomplexity;
    code file="C:\Users\Mentesnot\OneDrive\Desktop\Data Science Metro\SAS project\treeoutput.sas";  /* Save scoring logic */
    output out=scored;  /* Output scored dataset */
run;
quit;

/* Turn off ODS Graphics */
/*ods graphics off;*/

/* Score the test data */
data tree_test_score;
    set testdata;  /* Use test data */
    %include 'C:\Users\Mentesnot\OneDrive\Desktop\Data Science Metro\SAS project\treeoutput.sas';  /* Apply tree scoring */
run;




/************************** Generate ROC Curves for All Models (Logistic, GAM, Tree) ****************************/
ods graphics on;
proc logistic data=test_pred_comp;
    where phatglm ne . and phatgam ne . and Phattree ne .; /* Filter out missing predictions */
    class CombinedRisk AgeGroup TempCategory HeartRateCategory; /* Update classes for your dataset */
    model CombinedRisk(event="At Risk")= ; 
    roc 'Logistic' pred=phatglm;
    roc 'AUC for GAM' pred=phatgam;
    roc 'AUC for Tree' pred=Phattree;
run;
ods graphics off;

/*********************************** Sampling for Survey Logistic Regression ***********************************/
proc surveyselect data=MaternalHealthCategorized1 rate=0.7 outall out=rand_maternalhealth seed=123;
run;

proc freq data=rand_maternalhealth;
    table Selected;
run;
quit;

/************************************ Survey Logistic: Test Sample ************************************/
proc surveylogistic data=rand_maternalhealth;
    domain selected;
    class AgeGroup TempCategory HeartRateCategory; /* Update classes */
    model CombinedRisk(event="At Risk")= AgeGroup TempCategory HeartRateCategory SystolicBP DiastolicBP BS BodyTemp HeartRate; /* Adjust model variables */
    output out=surveyout(where=(selected=0)) p=phat; /* Only output for test sample */
run;

proc logistic data=surveyout;
    where phat ne .;
    class AgeGroup TempCategory HeartRateCategory; /* Same classes */
    baseline_model:model CombinedRisk(event="At Risk")= ; 
    roc 'AUC of the Hold Out Sample' pred=phat;
run;

/************************************ Survey Logistic: Training Sample ************************************/
proc surveylogistic data=rand_maternalhealth;
    domain selected;
    class AgeGroup TempCategory HeartRateCategory; /* Same classes */
    model CombinedRisk(event="At Risk")= AgeGroup TempCategory HeartRateCategory SystolicBP DiastolicBP BS BodyTemp HeartRate;
    output out=survetrain(where=(selected=1)) p=phat;
run;

proc logistic data=survetrain;
    where phat ne .;
    class AgeGroup TempCategory HeartRateCategory;
    baseline_model:model CombinedRisk(event="At Risk")= ;
    roc 'AUC of Train Sample' pred=phat;
run;

/************************************ LASSO Regression with HPGENSELECT ************************************/
ods html;
ods graphics on;
proc hpgenselect data=traindata;
    class AgeGroup(ref="40 and O");
    model CombinedRisk(event="At Risk") = AgeGroup BP_Interaction Age_BP_Interaction Log_SystolicBP Log_Age Log_DiastolicBP Log_BS Log_BodyTemp Log_HeartRate;/ distribution=binary cl;
    selection method=LASSO(choose=SBC) details=all;
    output out=out1 p=predlasso;
run;
ods graphics off;
ods html close;



ods html;
ods graphics on;

proc hpgenselect data=traindata;
    class AgeGroup(ref="40 and O");
    model CombinedRisk(event="At Risk") = AgeGroup 
                                           BP_Interaction 
                                           Age_BP_Interaction 
                                           Log_SystolicBP 
                                           Log_Age 
                                           Log_DiastolicBP 
                                           Log_BS 
                                           Log_BodyTemp 
                                           Log_HeartRate / 
                                           distribution=binary cl;
    selection method=LASSO(choose=SBC) details=all;
    output out=out1 p=predlasso; /* Output dataset with predicted values */
run;

ods graphics off;
ods html close;
