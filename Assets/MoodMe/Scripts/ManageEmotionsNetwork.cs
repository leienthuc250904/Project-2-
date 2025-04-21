using UnityEngine;
using Unity.Sentis;
using System;
using System.Collections.Generic;
using System.Linq;


namespace MoodMe
{
    public class ManageEmotionsNetwork : MonoBehaviour
    {
        public ModelAsset EmotionsNetwork;
        public int ImageNetworkWidth = 224; //48
        public int ImageNetworkHeight = 224; //48
        [Range(1, 4)] public int ChannelCount = 1;
        public bool Process;

        private Queue<Tensor<float>> frameBuffer = new Queue<Tensor<float>>();

        public float[] GetCurrentEmotionValues
        {
            get { return DetectedEmotions.Values.ToArray(); }
        }

        private Worker worker;
        private Model runtimeModel;
        private static Dictionary<string, float> DetectedEmotions;
        private string[] EmotionsLabelFull = { "Angry", "Disgusted", "Scared", "Happy", "Sad", "Surprised", "Neutral" };
        private string[] EmotionsLabel = { "Neutral", "Surprised", "Sad" }; //Free Package


        void Start()
        {
            //------------NEW-------------------
            var sourceModel = ModelLoader.Load(EmotionsNetwork);
            Debug.Log($"Model input shape: {sourceModel.inputs[0].shape}");
            Debug.Log($"Model input shape: {sourceModel.inputs[0].GetType()}");

            // FunctionalGraph graph = new FunctionalGraph();
            // FunctionalTensor inputs = graph.AddInput<float>(new TensorShape(1, 10, 1, ImageNetworkWidth, ImageNetworkHeight));
            // FunctionalTensor[] tensorsToStack = new FunctionalTensor[10];

            // // Fill the array with the same tensor 10 times
            // // We need to clone the tensor for each frame to make proper FunctionalTensors
            // for (int i = 0; i < 10; i++)
            // {
            //     // Create a functional tensor from the input tensor
            //     tensorsToStack[i] = Functional.Clone(inputs);
            // }

            // // Stack the tensors along dimension 1 (creating the time/frame dimension)
            // FunctionalTensor stackedTensor = Functional.Stack(tensorsToStack, dim: 1);
            // FunctionalTensor[] outputs = Functional.Forward(sourceModel, stackedTensor);
            // runtimeModel = graph.Compile(outputs);
            // FunctionalGraph graph = new FunctionalGraph();

            // // Create input with the correct shape
            // FunctionalTensor input = graph.AddInput(DataType.Float, new TensorShape(1, 10, 1, ImageNetworkWidth, ImageNetworkHeight));

            // // Forward the input through the source model
            // var outputs = Functional.Forward(sourceModel, input);

            // // Compile the modified model
            // runtimeModel = graph.Compile(outputs);

            // if (runtimeModel == null)
            // {
            //     Debug.LogError("Failed to compile runtime model");
            //     return;
            // }


            //-----------------------------------
            //var runtimeModel = ModelLoader.Load(EmotionsNetwork);
            worker = new Worker(sourceModel, BackendType.GPUCompute);

            // Initialize DetectedEmotions dictionary
            DetectedEmotions = new Dictionary<string, float>();
            foreach (string key in EmotionsLabelFull)
            {
                DetectedEmotions.Add(key, 0);
            }
        }

        void Update()
        {
            if (!Process)
            {
                Debug.Log("Processing not enabled, skipping Update.");
                return;
            }
            Process = true;


            // if (FaceDetection.OutputCrop == null || FaceDetection.OutputCrop.Length != (ImageNetworkWidth * ImageNetworkHeight))
            // {
            //     Debug.Log("running " + FaceDetection.IsFaceDetectionRunning);
            //     if (FaceDetection.IsFaceDetectionRunning)
            //     {
            //         Debug.LogWarning("OutputCrop is null or has incorrect dimensions, skipping processing.");
            //     }
            //     return;
            // }
            //DEBUG--
            // Check if FaceDetection is available
            // if (FaceDetection == null)
            // {
            //     Debug.LogError("FaceDetection is null. Please ensure FaceDetection component is properly referenced.");
            //     return;
            // }

            // Check if OutputCrop exists
            Debug.Log("Update");
            if (FaceDetection.OutputCrop == null)
            {
                Debug.Log("FaceDetection.OutputCrop is null");
                return;
            }

            // Check dimensions
            if (FaceDetection.OutputCrop.Length != (ImageNetworkWidth * ImageNetworkHeight))
            {
                Debug.LogWarning($"OutputCrop has incorrect dimensions. Expected: {ImageNetworkWidth * ImageNetworkHeight}, Got: {FaceDetection.OutputCrop.Length}");
                return;
            }

            Debug.Log("Output length" + FaceDetection.OutputCrop.Length);


            // Create the input texture and assign pixel data
            Texture2D croppedTexture = null;
            // Tensor<float> finalInputTensor = null;
            try
            {
                // List<Tensor<float>> frameBuffer = new List<Tensor<float>>();

                croppedTexture = new Texture2D(ImageNetworkWidth, ImageNetworkHeight, TextureFormat.R8, false);
                Color32[] rgba = FaceDetection.OutputCrop;
                croppedTexture.SetPixels32(rgba);
                croppedTexture.Apply();

                // Convert texture to tensor with NHWC layout
                var transform = new TextureTransform().SetTensorLayout(TensorLayout.NHWC).SetDimensions(ImageNetworkWidth, ImageNetworkHeight, ChannelCount); //NHWC
                using (var inputTensor = TextureConverter.ToTensor(croppedTexture, transform))
                {
                    //NEW

                    // FunctionalTensor[] tensorsToStack = new FunctionalTensor[10];

                    // // Fill the array with the same tensor 10 times
                    // // We need to clone the tensor for each frame to make proper FunctionalTensors
                    // for (int i = 0; i < 10; i++)
                    // {
                    //     // Create a functional tensor from the input tensor
                    //     tensorsToStack[i] = Functional.Clone(inputTensor);
                    // }

                    // // Stack the tensors along dimension 1 (creating the time/frame dimension)
                    // FunctionalTensor stackedTensor = Functional.Stack(tensorsToStack, dim: 1);
                    frameBuffer.Enqueue(inputTensor.ReadbackAndClone());

                    // Keep only the last 10 frames
                    if (frameBuffer.Count > 10)
                    {
                        var oldFrame = frameBuffer.Dequeue();
                        oldFrame.Dispose();
                    }

                    // Process when we have enough frames
                    if (frameBuffer.Count == 10)
                    {
                        // Create stacked tensor
                        var stackedShape = new TensorShape(1, 10, 1, ImageNetworkWidth, ImageNetworkHeight);
                        var stackedTensor = new Tensor<float>(stackedShape);

                        int frameIndex = 0;
                        foreach (var frame in frameBuffer)
                        {
                            // Copy each frame into the appropriate position in the stacked tensor
                            for (int h = 0; h < ImageNetworkHeight; h++)
                            {
                                for (int w = 0; w < ImageNetworkWidth; w++)
                                {
                                    stackedTensor[0, frameIndex, 0, h, w] = frame[0, h, w, 0];
                                }
                            }
                            frameIndex++;
                        }





                        Debug.Log("Input tensor shape: " + stackedTensor.shape);

                        worker.Schedule(stackedTensor); //inputTensor

                        using (var outputTensor = worker.PeekOutput() as Tensor<float>)
                        {
                            if (outputTensor == null)
                            {
                                Debug.LogError("Output tensor is null. Model inference failed.");
                                return;
                            }

                            using (var clonedTensor = outputTensor.ReadbackAndClone())
                            {

                                float[] results = clonedTensor.AsReadOnlyNativeArray().ToArray();
                                Debug.Log("Emotion result" + results);
                                for (int i = 0; i < results.Length; i++)
                                {
                                    if (i < EmotionsLabel.Length)
                                    {
                                        // DetectedEmotions[EmotionsLabel[i]] = results[i];
                                        DetectedEmotions[EmotionsLabelFull[i]] = results[i];
                                    }
                                }
                            }
                        }

                    }


                }
            }
            finally
            {
                if (croppedTexture != null)
                {
                    DisposeTexture(croppedTexture);
                }

            }
        }

        private void OnDisable()
        {
            worker?.Dispose();
            worker = null;
            Debug.Log("Disposed of worker on disable.");
        }

        private void OnDestroy()
        {
            worker?.Dispose();
            Debug.Log("Worker disposed on destroy.");
        }

        private void DisposeTexture(Texture2D texture)
        {
            if (texture != null)
            {
                Destroy(texture);
                texture = null;
            }
        }
    }
}