#include <stdbool.h>
#include "DataLoader.h"

int main(){
    char dataPath[] = "../Data/Animals/";
    int numOfImages = getNumberOfImages(dataPath);
    struct Image images[numOfImages];
    loadDataFromClasses(dataPath, images, true);
}
