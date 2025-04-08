#ifdef DATALOADER_EXPORTS 
  #define DATALOADER_API __declspec(dllexport)
#else
  #define DATALOADER_API __declspec(dllimport)
#endif

#define CALL __cdecl

#ifdef __cplusplus
extern "C"
{
#endif

#define MAX_LABEL_LENGTH  50
DATALOADER_API struct Image{
    int width;
    int height;
    int channels;
    char name[MAX_LABEL_LENGTH];
    char label[MAX_LABEL_LENGTH];
    unsigned char* values;
};

DATALOADER_API void CALL loadDataFromClasses(char dataPath[], struct Image* images, bool shuffle);
DATALOADER_API int CALL getNumberOfImages(char dirPath[]);
DATALOADER_API void CALL getImagesFromDir(char dataPath[], char className[], int* index, struct Image* images);
DATALOADER_API void CALL splitData(
               struct Image* images, int numOfimages,
               struct Image* trainImages, int numOfTrainImages,
               struct Image* testImages, int numOfTestImages,
               struct Image* validationImages, int numOfValidationImages,
               float* percentages
              );
DATALOADER_API void CALL shuffleImages(struct Image* images, int numOfImages);
DATALOADER_API void CALL swap (struct Image* a, struct Image* b);
DATALOADER_API void CALL checkPercentages(float* percentages);
DATALOADER_API void CALL loadNumOfImagesForSplits(int* numOfTrainImages, int* numOfTestImages, int* numOfValidationImages, float* percentages, int numOfImages);

#ifdef __cplusplus
} 
#endif