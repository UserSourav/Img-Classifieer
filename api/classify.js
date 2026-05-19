// api/classify.js
// Vercel serverless function — runs on the server, never exposed to browser
// Your Google Vision API key lives here via environment variable

export default async function handler(req, res) {
  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { imageBase64 } = req.body;

  if (!imageBase64) {
    return res.status(400).json({ error: 'Missing imageBase64 in request body' });
  }

  const apiKey = process.env.GOOGLE_VISION_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'API key not configured on server' });
  }

  try {
    const visionRes = await fetch(
      `https://vision.googleapis.com/v1/images:annotate?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          requests: [
            {
              image: { content: imageBase64 },
              features: [
                { type: 'LABEL_DETECTION',         maxResults: 10 },
                { type: 'OBJECT_LOCALIZATION',     maxResults: 5  },
                { type: 'IMAGE_PROPERTIES',        maxResults: 1  },
              ],
            },
          ],
        }),
      }
    );

    if (!visionRes.ok) {
      const errText = await visionRes.text();
      console.error('Google Vision error:', errText);
      return res.status(502).json({ error: 'Google Vision API request failed' });
    }

    const visionData = await visionRes.json();
    const response = visionData.responses?.[0];

    if (!response) {
      return res.status(500).json({ error: 'Empty response from Google Vision' });
    }

    // Parse labels
    const labels = (response.labelAnnotations || []).map(l => ({
      label: l.description,
      confidence: parseFloat((l.score * 100).toFixed(1)),
    }));

    // Parse detected objects (more specific than labels)
    const objects = (response.localizedObjectAnnotations || []).map(o => ({
      label: o.name,
      confidence: parseFloat((o.score * 100).toFixed(1)),
    }));

    return res.status(200).json({
      labels,
      objects,
      source: 'google-vision',
    });

  } catch (err) {
    console.error('Server error:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
