using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DrawSkeleton : MonoBehaviour
{

	public GameObject[] keypoints;
	private GameObject[] lines;
	private LineRenderer[] lineRenderers;
	private int[][] jointPairs;
	private float lineWidth = 5.0f;

    // Start is called before the first frame update
    void Start()
    {
		int numPairs = keypoints.Length + 1;
		lines = new GameObject[numPairs];
		lineRenderers = new LineRenderer[numPairs];
		jointPairs = new int[numPairs][];

		InitializeSkeleton();

    }

    // Update is called once per frame
    void Update()
    {

    }

	void LateUpdate() {
		RenderSkeleton();
	}


	private void InitializeLine(int pairIndex, int startIndex, int endIndex, float width, Color color) {
        jointPairs[pairIndex] = new int[] { startIndex, endIndex };
        string name = $"{keypoints[startIndex].name}_to_{keypoints[endIndex].name}";
        lines[pairIndex] = new GameObject(name);

        lineRenderers[pairIndex] = lines[pairIndex].AddComponent<LineRenderer>();
        lineRenderers[pairIndex].material = new Material(Shader.Find("Unlit/Color"));
        lineRenderers[pairIndex].material.color = color;

        lineRenderers[pairIndex].positionCount = 2;
        lineRenderers[pairIndex].startWidth = width;
        lineRenderers[pairIndex].endWidth = width;
    }

	private void InitializeSkeleton() {
        // Nose to left eye
        InitializeLine(0, 0, 1, lineWidth, Color.magenta);
        // Nose to right eye
        InitializeLine(1, 0, 2, lineWidth, Color.magenta);
        // Left eye to left ear
        InitializeLine(2, 1, 3, lineWidth, Color.magenta);
        // Right eye to right ear
        InitializeLine(3, 2, 4, lineWidth, Color.magenta);

        // Left shoulder to right shoulder
        InitializeLine(4, 5, 6, lineWidth, Color.red);
        // Left shoulder to left hip
        InitializeLine(5, 5, 11, lineWidth, Color.red);
        // Right shoulder to right hip
        InitializeLine(6, 6, 12, lineWidth, Color.red);
        // Left shoulder to right hip
        InitializeLine(7, 5, 12, lineWidth, Color.red);
        // Right shoulder to left hip
        InitializeLine(8, 6, 11, lineWidth, Color.red);
        // Left hip to right hip
        InitializeLine(9, 11, 12, lineWidth, Color.red);

        // Left Arm
        InitializeLine(10, 5, 7, lineWidth, Color.green);
        InitializeLine(11, 7, 9, lineWidth, Color.green);
        // Right Arm
        InitializeLine(12, 6, 8, lineWidth, Color.green);
        InitializeLine(13, 8, 10, lineWidth, Color.green);

        // Left Leg
        InitializeLine(14, 11, 13, lineWidth, Color.blue);
        InitializeLine(15, 13, 15, lineWidth, Color.blue);
        // Right Leg
        InitializeLine(16, 12, 14, lineWidth, Color.blue);
        InitializeLine(17, 14, 16, lineWidth, Color.blue);
    }

	private void RenderSkeleton() {
        for (int i = 0; i < jointPairs.Length; i++)
        {
            int startpointIndex = jointPairs[i][0];
            int endpointIndex = jointPairs[i][1];

            GameObject startingKeyPoint = keypoints[startpointIndex];
            GameObject endingKeyPoint = keypoints[endpointIndex];

            Vector3 startPos = new Vector3(startingKeyPoint.transform.position.x,
                                           startingKeyPoint.transform.position.y,
                                           startingKeyPoint.transform.position.z);
            Vector3 endPos = new Vector3(endingKeyPoint.transform.position.x,
                                         endingKeyPoint.transform.position.y,
                                         endingKeyPoint.transform.position.z);

            if (startingKeyPoint.activeInHierarchy && endingKeyPoint.activeInHierarchy) {
                lineRenderers[i].gameObject.SetActive(true);
                lineRenderers[i].SetPosition(0, startPos);
                lineRenderers[i].SetPosition(1, endPos);
            } else {
                lineRenderers[i].gameObject.SetActive(false);
            }
        }
    }


}
