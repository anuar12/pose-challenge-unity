using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Video;

namespace Pose.Detection
{
	/// <summary>
	/// Pose Detection using Unity Barracuda.
	/// 1. Use an existing pose detection model (HRNet, DeepPose etc) and convert it to onnx model.
	/// 2. Render video to a RenderTarget and use it as the input for the model.
	/// 3. Preprocess the render target before passing it to the model. You can use Compute Shaders for this.
	/// 4. Each model requires a specific color space and resolution, you need to convert the render target first.
	/// 5. The `Model` object along with `IWorker` engine classes from Barracuda package will help you with inference.
	/// 6. Convert the output key points with confidence to actual pose and joint values.
	/// 7. Drive a skeleton/gameobjects with joint values.
	/// </summary>
	public class PoseDetection : MonoBehaviour
	{
		[SerializeField] private VideoPlayer videoPlayer;
		[SerializeField] private ComputeShader preprocessingShader;
		public RenderTexture videoTexture;

		[Space]
		[SerializeField] private int imageHeight = 352;
		[SerializeField] private int imageWidth = 352;

		[Space]
		[SerializeField] private GameObject videoQuad;

		public bool displayInput = false;
		public GameObject inputScreen;
		public RenderTexture inputTexture;


		#region private
		private int _videoHeight;
		private int _videoWidth;

		private const int numKeypoints = 17;
		// Estimated 2D keypoint locations in videoTexture and their associated confidence values
		private float[][] keypointLocations = new float[numKeypoints][];

		//Shader Property
		private static readonly int MainTex = Shader.PropertyToID("_MainTex");
		#endregion

		public NNModel modelAsset;
		public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

		private Model m_RuntimeModel;
		private IWorker engine;

		private string heatmapLayer = "heatmaps";

		public GameObject[] keypoints;

		[Range(0, 100)]
		public int minConfidence = 70;

		private float[][] prevPos = new float[numKeypoints][];
		private float[][] prevVelocity = new float[numKeypoints][];

		private void Start()
		{
			_videoHeight = (int)videoPlayer.GetComponent<VideoPlayer>().height;
			_videoWidth = (int)videoPlayer.GetComponent<VideoPlayer>().width;

			// Create a new videoTexture using the current video dimensions
			videoTexture = new RenderTexture(_videoWidth, _videoHeight, 24, RenderTextureFormat.ARGB32);
			videoPlayer.GetComponent<VideoPlayer>().targetTexture = videoTexture;

			//Apply Texture to Quad
			videoQuad.gameObject.GetComponent<MeshRenderer>().material.SetTexture(MainTex, videoTexture);
			videoQuad.transform.localScale = new Vector3(_videoWidth, _videoHeight, videoQuad.transform.localScale.z);
			videoQuad.transform.position = new Vector3(_videoWidth / 2, _videoHeight / 2, 1);

			//Move Camera to keep Quad in view
			var mainCamera = Camera.main;
			if (mainCamera != null)
			{
				mainCamera.transform.position = new Vector3(_videoWidth / 2, _videoHeight / 2, -(_videoWidth / 2));
				mainCamera.GetComponent<Camera>().orthographicSize = _videoHeight / 2;
			}

			m_RuntimeModel = ModelLoader.Load(modelAsset);

			var modelBuilder = new ModelBuilder(m_RuntimeModel);

			engine = WorkerFactory.CreateWorker(workerType, modelBuilder.model);
		}


		// Unity method that is called every tick/frame
		private void Update()
		{
			Texture2D processedImage = PreprocessTexture();

			if (displayInput)
			{
				inputScreen.SetActive(true);
				// Graphics.Blit(processedImage, inputTexture);
				Texture2D scaledInputImage = ScaleInputImage(processedImage);
				Graphics.Blit(scaledInputImage, inputTexture);
				Destroy(scaledInputImage);
			}
			else
			{
				inputScreen.SetActive(false);
			}

			Tensor input = new Tensor(processedImage, channels: 3);

			engine.Execute(input);

			ProcessResults(engine.PeekOutput(heatmapLayer));

			UpdateKeyPointPositions();
			FillAndUpdatePrevPos();

			input.Dispose();

			Destroy(processedImage);
		}

		private void OnDisable()
		{
			engine.Dispose();

			//Release videoTexture
			videoTexture.Release();
		}


		#region Additional Methods

		private Texture2D PreprocessTexture()
		{
			//Apply any kind of preprocessing if required - Resize, Color values scaled etc

			Texture2D imageTexture = new Texture2D(videoTexture.width,
				videoTexture.height, TextureFormat.RGBA32, false);

			Graphics.CopyTexture(videoTexture, imageTexture);
			Texture2D tempTex = Resize(imageTexture, imageHeight, imageWidth);
			Destroy(imageTexture);

			imageTexture = PreprocessNetwork(tempTex);

			Destroy(tempTex);
			return imageTexture;
		}

		private Texture2D Resize(Texture2D image, int newWidth, int newHeight)
		{
			RenderTexture rTex = RenderTexture.GetTemporary(newWidth, newHeight, 24);
			RenderTexture.active = rTex;

			Graphics.Blit(image, rTex);
			Texture2D nTex = new Texture2D(newWidth, newHeight, TextureFormat.RGBA32, false);

			Graphics.CopyTexture(rTex, nTex);
			RenderTexture.active = null;

			RenderTexture.ReleaseTemporary(rTex);
			return nTex;
		}

		private Texture2D PreprocessNetwork(Texture2D inputImage)
		{
			// Use Compute Shaders (GPU) to preprocess your image
			// Each model requires a specific color space - RGB
			// Values need to scaled to what it was trained on

			var numthreads = 8;
			var kernelHandle = preprocessingShader.FindKernel("Standardize");
			var rTex = new RenderTexture(inputImage.width,
				inputImage.height, 24, RenderTextureFormat.ARGBHalf);
			rTex.enableRandomWrite = true;
			rTex.Create();

			preprocessingShader.SetTexture(kernelHandle, "Result", rTex);
			preprocessingShader.SetTexture(kernelHandle, "InputImage", inputImage);
			preprocessingShader.Dispatch(kernelHandle, inputImage.height
													   / numthreads,
				inputImage.width / numthreads, 1);

			RenderTexture.active = rTex;
			Texture2D nTex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGBAHalf, false);
			Graphics.CopyTexture(rTex, nTex);
			RenderTexture.active = null;

			Destroy(rTex);
			return nTex;
		}

		private void ProcessResults(Tensor heatmaps)
		{
			float stride = imageHeight / heatmaps.shape.height;

			var scaleY = (float)videoTexture.height / (float)imageHeight;
			var scaleX = (float)videoTexture.width / (float)imageWidth;

			for (int k = 0; k < numKeypoints; k++)
			{
				var locationInfo = LocateKeyPoint(heatmaps, k);
				var coords = locationInfo.Item1;
				var confidenceValue = locationInfo.Item2;

				var xPos = coords[0] * stride * scaleX;
				var yPos = (imageHeight - (coords[1] * stride)) * scaleY;

				keypointLocations[k] = new float[] { xPos, yPos, confidenceValue };
			}
		}

		private (float[], float) LocateKeyPoint(Tensor heatmaps, int i)
		{
			//Find the heatmap index that contains the highest confidence value and the associated offset vector
			var maxConfidence = 0f;
			var coords = new float[2];

			for (int y = 0; y < heatmaps.shape.height; y++)
			{
				for (int x = 0; x < heatmaps.shape.width; x++)
				{
					if (heatmaps[0, y, x, i] > maxConfidence)
					{
						maxConfidence = heatmaps[0, y, x, i];
						coords = new float[] { x, y };
					}
				}
			}
			return (coords, maxConfidence);
		}

		private Texture2D ScaleInputImage(Texture2D inputImage)
		{
			// Specify the number of threads on the GPU
			int numthreads = 8;
			// Get the index for the ScaleInputImage function in the ComputeShader
			int kernelHandle = preprocessingShader.FindKernel("ScaleInputImage");
			// Define an HDR RenderTexture
			RenderTexture rTex = new RenderTexture(inputImage.width, inputImage.height, 24, RenderTextureFormat.ARGBHalf);
			// Enable random write access
			rTex.enableRandomWrite = true;
			// Create the HDR RenderTexture
			rTex.Create();

			// Set the value for the Result variable in the ComputeShader
			preprocessingShader.SetTexture(kernelHandle, "Result", rTex);
			// Set the value for the InputImage variable in the ComputeShader
			preprocessingShader.SetTexture(kernelHandle, "InputImage", inputImage);

			// Execute the ComputeShader
			preprocessingShader.Dispatch(kernelHandle, inputImage.height / numthreads, inputImage.width / numthreads, 1);
			// Make the HDR RenderTexture the active RenderTexture
			RenderTexture.active = rTex;

			// Create a new HDR Texture2D
			Texture2D nTex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGBAHalf, false);

			// Copy the RenderTexture to the new Texture2D
			Graphics.CopyTexture(rTex, nTex);
			// Make the HDR RenderTexture not the active RenderTexture
			RenderTexture.active = null;
			// Remove the HDR RenderTexture
			Destroy(rTex);
			return nTex;
		}

		private void UpdateKeyPointPositions()
		{
			for (int k = 0; k < numKeypoints; k++)
			{
				if (keypointLocations[k][2] >= minConfidence / 100f)
				{
					keypoints[k].SetActive(true);
					//keypoints[k].activeInHierarchy
				}
				else
				{
					keypoints[k].SetActive(false);
				}

				Vector3 newPos = new Vector3(keypointLocations[k][0], keypointLocations[k][1], -1f);
				keypoints[k].transform.position = newPos;
			}
		}

		// Simple extrapolation of keypoints when the current frame's joint keypoints
		// is not available. If not available: use prevPos + prevVelocity position.
		private void FillAndUpdatePrevPos()
		{
			for (int k = 0; k < numKeypoints; k++)
			{
				if (keypointLocations[k][2] < minConfidence / 100f)
				{
					// Fill in the keypoints if possible
					if (prevPos[k] != null && prevPos[k][2] >= minConfidence / 100f)
					{
						Vector3 newPos;
						if (prevVelocity[k] != null)
						{
							newPos = new Vector3(prevPos[k][0], prevPos[k][1], -1f);
						}
						else
						{
							newPos = new Vector3(prevPos[k][0] + prevVelocity[k][0],
								prevPos[k][1] + prevVelocity[k][1],
								-1f);
						}
						keypoints[k].transform.position = newPos;
						keypoints[k].SetActive(true);
					}
					prevPos[k] = new float[] { 0, 0, -1 };
					prevVelocity[k] = new float[] { 0, 0, -1 };
				}
				else
				{  // update the prev positions and velocities
					if (prevPos[k] != null)
					{
						prevVelocity[k] = new float[] {
							keypointLocations[k][0] - prevPos[k][0],
							keypointLocations[k][1] - prevPos[k][1],
							1 };
					}
					else
					{
						prevVelocity[k] = new float[] { 0, 0, -1 };
					}
					prevPos[k] = keypointLocations[k];
				}
			}
		}

		#endregion
	}
}