weather <- read.csv("C:/Users/Rajan Kumar Gupta/Downloads/weatherHistory.csv")
View(weather)
str(weather)

#Preprocessing and cleaning data
weather<-weather[, c("Precip.Type","Temperature..C.","Humidity","Wind.Speed..km.h.")]
weather$Precip.Type[weather$Precip.Type=="null"]<-"Sunny"
View(weather)
sum(duplicated(weather))
weather<-weather[!duplicated(weather),]
sum(duplicated(weather))
weather<-na.omit(weather)
cleaned_data<-weather
table(cleaned_data$Precip.Type)

#Normalizing the data using scale function
cleaned_data$Temperature..C.<-scale(cleaned_data$Temperature..C.)
cleaned_data$Humidity<-scale(cleaned_data$Humidity)
cleaned_data$Wind.Speed..km.h.<-scale(cleaned_data$Wind.Speed..km.h.)
cleaned_data$Precip.Type<-as.factor(cleaned_data$Precip.Type)

#Paritation of data
library(caret)
index<-createDataPartition(cleaned_data$Precip.Type, p=0.8,list=FALSE)
training_set<-cleaned_data[index,]
testing_set<-cleaned_data[-index,]

training_labels<-cleaned_data[index,"Precip.Type"]
testing_labels<-cleaned_data[-index,"Precip.Type"]

#Evaluating data using KNN Algorithm
library(class)
knn_model<-knn(training_set[-1],testing_set[-1],cl=training_labels,k=3)
confus_mat<-confusionMatrix(knn_model, testing_labels)
print(confus_mat)

#Evaluating data using Naive Bayes Algorithm
library(e1071)
naive_bayes<-naiveBayes(Precip.Type~.,data=training_set)
prediction<-predict(naive_bayes,testing_set)
confus_mat1<-confusionMatrix(prediction,testing_labels)
print(confus_mat1)

#Evaluating data using decision tree algorithm
library(rpart)
library(rpart.plot)
library(caret)
decision_tree<-rpart(Precip.Type~.,data=training_set,method="class")
rpart.plot(decision_tree)
pre1<-predict(decision_tree,testing_set,type = "class")
confus_mat2<-confusionMatrix(pre1,testing_labels)
print(confus_mat2)

#Evaluating data using Support Vector Machine Algorithm
library(e1071)
library(caret)
svm_model<-svm(Precip.Type~.,data=training_set,kernel="linear")
pre2<-predict(svm_model,testing_set)
confus_mat3<-confusionMatrix(pre2,testing_labels)
print(confus_mat3)

#Comparing all Algorithm
knn_accuracy<-confus_mat$overall["Accuracy"]
nb_accuracy<-confus_mat1$overall["Accuracy"]
dt_accuracy<-confus_mat2$overall["Accuracy"]
svm_accuracy<-confus_mat3$overall["Accuracy"]

cat("__MODEL ACCURACY__\n")
cat("KNN ACCURACY: ",round(knn_accuracy*100,2),"%\n")
cat("NAIVE BAYES ACCURACY: ",round(nb_accuracy*100,2),"%\n")
cat("DECISION TREE ACCURACY: ",round(dt_accuracy*100,2),"%\n")
cat("SUPPORT VECTOR MACHINE ACCURACY: ",round(svm_accuracy*100,2),"%\n")

cat("__KNN PREDICTION__")
print(table(knn_model,testing_labels))

cat("__NAIVE BAYES PREDICTION__")
print(table(prediction,testing_labels))

cat("__DECISION TREE PREDICTION__")
print(table(pre1,testing_labels))

cat("__SUPPORT VECTOR MACHINE PREDICTION__")
print(table(pre2,testing_labels))

#Visualizing both Prediction and Accurary of all Algorithm
library(ggplot2)
accuracy_df<-data.frame(
  Model=c("KNN","Naive Bayes","Decision Tree","SVM"),
  Accuracy=c(knn_accuracy*100,nb_accuracy*100,dt_accuracy*100,svm_accuracy*100)
  )
accuracy_comparision_plot<-ggplot(accuracy_df,aes(x=Model,y=Accuracy,fill=Model))+
                                    geom_bar(stat = "identity",width=0.6)+
                                    ylim(0,100)+
                                    labs(title="Model Accuracy Comparision", y="Accuracy(%)",x="Model")+
                                    theme_minimal()+
                                    scale_fill_manual(values = c("KNN"="green","Naive Bayes"="blue","Decision Tree"="cyan","SVM"="pink"))+
                                    geom_text(aes(label=round(Accuracy, 2)),vjust=-0.5)
print(accuracy_comparision_plot)

#Comparision table of Predicted and Actual Precipitation type
knn_table<-as.data.frame(table(Predicted=knn_model,Actual=testing_labels))
nb_table<-as.data.frame(table(Predicted=prediction,Actual=testing_labels))
dt_table<-as.data.frame(table(Predicted=pre1,Actual=testing_labels))
svm_table<-as.data.frame(table(Predicted=pre2,Actual=testing_labels))
print(knn_table)
print(nb_table)
print(dt_table)
print(svm_table)

#KNN Algorithm Confusion Matrix Plot
knn_confMatrix<-ggplot(knn_table,aes(x=Actual,y=Predicted,fill=Freq))+
  geom_tile()+
  geom_text(aes(label=Freq),color="black")+
  scale_fill_gradient(low="white",high="cyan")+
  labs(title="KNN Confusion Matrix",x="Actual",y="Predicted")
print(knn_confMatrix)

#Navive Bayes Confusion Matrix Plot
nb_confMatrix<-ggplot(nb_table,aes(x=Actual,y=Predicted,fill=Freq))+
  geom_tile()+
  geom_text(aes(label=Freq),color="black")+
  scale_fill_gradient(low="white",high="orange")+
  labs(title="Naive Bayes Confusion Matrix",x="Actual",y="Predicted")
print(nb_confMatrix)

#Decision Tree Confusion Matrix Plot
dt_confMatrix<-ggplot(dt_table,aes(x=Actual,y=Predicted,fill=Freq))+
  geom_tile()+
  geom_text(aes(label=Freq),color="black")+
  scale_fill_gradient(low="white",high="blue")+
  labs(title="Decision Tree Confusion Matrix",x="Actual",y="Predicted")
print(dt_confMatrix)

#Support Vector Machine Confusion Matrix Plot
svm_confMatrix<-ggplot(svm_table,aes(x=Actual,y=Predicted,fill=Freq))+
  geom_tile()+
  geom_text(aes(label=Freq),color="black")+
  scale_fill_gradient(low="white",high="grey")+
  labs(title="Support Vector Machine Confusion Matrix",x="Actual",y="Predicted")
print(svm_confMatrix)

# Define UI for Shiny Dashboard
library(shiny)
ui <- fluidPage(
  titlePanel("Weather Prediction Dashboard"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("model_choice", 
                  "Choose a Model for Analysis:", 
                  choices = c("KNN", "Naive Bayes", "Decision Tree", "SVM")),
      actionButton("show_accuracy", "Show Accuracy"),
      actionButton("show_confusion_matrix", "Show Confusion Matrix")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Model Accuracy", plotOutput("accuracy_plot")),
        tabPanel("Confusion Matrix", plotOutput("conf_matrix_plot")),
        tabPanel("Prediction Comparison", tableOutput("comparison_table"))
      )
    )
  )
)

# Define Server Logic
server <- function(input, output) {
  # Plotting Model Accuracy
  output$accuracy_plot <- renderPlot({
    ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
      geom_bar(stat = "identity", width = 0.6) +
      ylim(0, 100) +
      labs(title = "Model Accuracy Comparison", y = "Accuracy (%)", x = "Model") +
      theme_minimal() +
      scale_fill_manual(values = c("KNN" = "green", "Naive Bayes" = "blue", 
                                   "Decision Tree" = "cyan", "SVM" = "pink")) +
      geom_text(aes(label = round(Accuracy, 2)), vjust = -0.5)
  })
  
  # Dynamic Confusion Matrix Plot
  output$conf_matrix_plot <- renderPlot({
    if (input$model_choice == "KNN") {
      knn_confMatrix
    } else if (input$model_choice == "Naive Bayes") {
      nb_confMatrix
    } else if (input$model_choice == "Decision Tree") {
      dt_confMatrix
    } else if (input$model_choice == "SVM") {
      svm_confMatrix
    }
  })
  
  # Prediction Comparison Table
  output$comparison_table <- renderTable({
    if (input$model_choice == "KNN") {
      knn_table
    } else if (input$model_choice == "Naive Bayes") {
      nb_table
    } else if (input$model_choice == "Decision Tree") {
      dt_table
    } else if (input$model_choice == "SVM") {
      svm_table
    }
  })
}

# Run the Shiny App
shinyApp(ui = ui, server = server)
