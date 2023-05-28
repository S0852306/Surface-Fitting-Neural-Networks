function Report=FittingReport(data,label,NN)
close all
prediction=NN.Evaluate(data);
error=label-prediction;
Report.ErrorVector=error;
Report.Prediction=prediction;
NumOfOutput=size(label,1);
if NumOfOutput==1
    Report.MeanAbsoluteError=mean(abs(error));
    figure;
    histogram(error)
    title('Error Distribution')
    figure;
    scatter(prediction,label)
    hold on
    range=[min(label),max(label)];
    plot(range,range,'LineWidth',2)
    hold off
    title('Predict v.s. Actual')
else
    Report.MeanAbsoluteError=mean(abs(error),2);
    figure;
    
    for i=1:NumOfOutput
        subplot(1,NumOfOutput,i)
        histogram(error(i,:))
        text=['Error Distribution, y' num2str(i)];
        title(text)
    end
    
    
    figure;
    for i=1:NumOfOutput
        subplot(1,NumOfOutput,i)
        scatter(prediction(i,:),label(i,:))
        hold on
        range=[min(label(i,:)),max(label(i,:))];
        plot(range,range,'LineWidth',2)
        hold off
        text=['Predict v.s. Actual, y' num2str(i)];
        title(text)
    end
    
end


end