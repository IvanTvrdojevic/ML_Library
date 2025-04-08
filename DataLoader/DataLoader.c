#include <stdio.h>
#include <dirent.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define MAX_LABEL_LENGTH  50

struct Image{
    int width;
    int height;
    int channels;
    char name[MAX_LABEL_LENGTH];
    char label[MAX_LABEL_LENGTH];
    unsigned char* values;
};

void loadDataFromClasses(char dataPath[], struct Image* images, bool shuffle, bool showInfo);
int getNumberOfImages(char dirPath[]);
void getImagesFromDir(char dataPath[], char className[], int* index, struct Image* images, bool showInfo);
void splitData(
               struct Image* images, int numOfimages,
               struct Image* trainImages, int numOfTrainImages,
               struct Image* testImages, int numOfTestImages,
               struct Image* validationImages, int numOfValidationImages,
               float* percentages
              );
void shuffleImages(struct Image* images, int numOfImages);
void swap (struct Image* a, struct Image* b);
void checkPercentages(float* percentages);
void loadNumOfImagesForSplits(int* numOfTrainImages, int* numOfTestImages, int* numOfValidationImages, float* percentages, int numOfImages);

int main(){
    char dataPath[] = "../Data/Animals/";
    int numOfImages = getNumberOfImages(dataPath);
    struct Image images[numOfImages];
    loadDataFromClasses(dataPath, images, true, false);

    float percentages[] = {70, 20, 10};
    int numOfTrainImages;
    int numOfTestImages;
    int numOfValidationImages;
    loadNumOfImagesForSplits(&numOfTrainImages, &numOfTestImages, &numOfValidationImages, percentages, numOfImages);

    struct Image trainImages[numOfImages];
    struct Image testImages[numOfTestImages];
    struct Image validationImages[numOfValidationImages];
    splitData(
              images, numOfImages, 
              trainImages, numOfTrainImages,
              testImages, numOfTestImages,
              validationImages, numOfValidationImages,
              percentages
             );

 
    int numOfTrainImagesFromSubdir = getNumberOfImages("../Data/SplitData/train");
    struct Image trainImagesFromSubdir[numOfTrainImagesFromSubdir];
    loadDataFromClasses("../Data/SplitData/train", trainImagesFromSubdir, true, false);

    int numOfTestImagesFromSubdir = getNumberOfImages("../Data/SplitData/test");
    struct Image testImagesFromSubdir[numOfTestImagesFromSubdir];
    loadDataFromClasses("../Data/SplitData/test", testImagesFromSubdir, true, false);

    int numOfValidationImagesFromSubdir = getNumberOfImages("../Data/SplitData/validation");
    struct Image validationImagesFromSubdir[numOfValidationImagesFromSubdir];
    loadDataFromClasses("../Data/SplitData/validation", validationImagesFromSubdir, true, false);
}

int getNumberOfImages(char dataPath[]){
    DIR *d;
    struct dirent *dir;
    d = opendir(dataPath);
    int cnt = 0;
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            if (!strcmp (dir->d_name, ".") || !strcmp (dir->d_name, "..")){
                continue;
            }
            
            int classPathLenght = strlen(dataPath) + strlen(dir->d_name) + 2;
            char classPath[classPathLenght];
            snprintf(classPath, classPathLenght, "%s%s%s", dataPath, dir->d_name, "/");
            DIR *dChild;
            struct dirent *dirChild;
            dChild = opendir(classPath);
            if (dChild) {
                while ((dirChild = readdir(dChild)) != NULL) {
                    if (!strcmp (dirChild->d_name, ".") || !strcmp (dirChild->d_name, "..")){
                        continue;
                    }
                    cnt++;
                }
                closedir(dChild);
            }
        }
        closedir(d);
    } 

    return cnt;
}

void loadDataFromClasses(char dataPath[], struct Image* images, bool shuffle, bool showInfo){
    int index = 0;
    DIR *d;
    struct dirent *dir;
    d = opendir(dataPath);
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            if (!strcmp (dir->d_name, ".") || !strcmp (dir->d_name, "..")){
                continue;
            }
            getImagesFromDir(dataPath, dir->d_name, &index, images, showInfo);
        }
        closedir(d);
    }
    if(shuffle)
        shuffleImages(images, getNumberOfImages(dataPath));

}

// Fisher–Yates shuffle Algorithm
void shuffleImages(struct Image* images, int numOfImages){
    srand ( time(NULL) );
    // Start from the last element and swap one by one. We don't
    // need to run for the first element that's why i > 0
    for (int i = numOfImages-1; i > 0; i--)
    {
        // Pick a random index from 0 to i
        int j = rand() % (i+1);
        // Swap arr[i] with the element at random index
        swap(&images[i], &images[j]);
    }
}

void swap(struct Image* a, struct Image* b){
    struct Image temp = *a;
    *a = *b;
    *b = temp;
}

void splitData(
               struct Image* images, int numOfimages,
               struct Image* trainImages, int numOfTrainImages,
               struct Image* testImages, int numOfTestImages,
               struct Image* validationImages, int numOfValidationImages,
               float* percentages
            )
{
    for(int i = 0; i < numOfimages; i++){   
        if(i < numOfTrainImages)
            trainImages[i] = images[i];
        else if (i < numOfTestImages + numOfTrainImages)
            testImages[i - numOfTrainImages] = images[i];     
        else
            validationImages[i - numOfTestImages - numOfTrainImages] = images[i];
        
    }
}

void loadNumOfImagesForSplits(int* numOfTrainImages, int* numOfTestImages, int* numOfValidationImages, float* percentages, int numOfImages){
    checkPercentages(percentages);
    *numOfTrainImages = (int)(numOfImages * (percentages[0]/100));
    *numOfTestImages = (int)(numOfImages * (percentages[1]/100));
    *numOfValidationImages = numOfImages - (*numOfTrainImages + *numOfTestImages);
}


void checkPercentages(float* percentages){
    if(percentages[0] + percentages[1] + percentages[2] != 100 || (percentages[0] == 0 || percentages[1] == 0, percentages[2] == 0)){
        printf("Error, sum of percentages for the train test validate split has to be 100 and every percentage has to be bigger than 0\n");
        printf("Current split:\ntrain: %f, test %f, validate: %f\n", percentages[0], percentages[1], percentages[2]);
        printf("Enter train set percetnage:\n");
        scanf("%f", &percentages[0]);
        printf("Enter test set percetnage:\n");
        scanf("%f", &percentages[1]);
        printf("Enter validation set percetnage:\n");
        scanf("%f", &percentages[2]);
        checkPercentages(percentages);
        }
    
    return ;
}


void getImagesFromDir(char dataPath[], char className[], int* index, struct Image* images, bool showInfo){
    int classPathLenght = strlen(dataPath) + strlen(className) + 2;
    char classPath[classPathLenght];
    snprintf(classPath, classPathLenght, "%s%s%s", dataPath, className, "/");

    int i = 0 + *index;
    DIR *d;
    struct dirent *dir;
    d = opendir(classPath);
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            if (!strcmp (dir->d_name, ".") || !strcmp (dir->d_name, "..")){
                continue;
            }

            int fullImagePathLength = strlen(classPath) + strlen(dir->d_name) + 1;
            char fullImagePath[fullImagePathLength];
            snprintf(fullImagePath, fullImagePathLength, "%s%s", classPath, dir->d_name);
            images[i].values = stbi_load(fullImagePath, &images[i].width, &images[i].height, &images[i].channels, 0);
            strncpy(images[i].name, dir->d_name, sizeof(images[i].name) - 1);
            images[i].name[sizeof(images[i].name) - 1] = '\0';
            strncpy(images[i].label, className, sizeof(images[i].label) - 1);
            images[i].name[sizeof(images[i].label) - 1] = '\0';
            if(showInfo){
                if(images[i].values == NULL)
                    printf("Error when loading image, file path: %s\n", fullImagePath);
                else{
                    printf("Loaded image with label: %s\n", images[i].label);
                    i++;
                }
            }  
            else
                if(images[i].values != NULL)
                    i++;
        }
        *index = i;
        closedir(d);
    } 
}
 