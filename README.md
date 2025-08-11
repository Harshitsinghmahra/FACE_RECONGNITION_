# FACE_RECONGNITION_
# Advanced Face Recognition System

A browser-based face recognition system using TensorFlow.js and face-api.js that allows you to:
- Collect face data through your webcam
- Train a recognition model
- Recognize faces in real-time
- Manage your dataset

## Features

- **Face Data Collection**: Capture multiple images of a person's face from different angles
- **Model Training**: Train a recognition model directly in your browser
- **Real-time Recognition**: Identify faces with confidence scores
- **Data Management**: View, manage, and clear your dataset
- **No Server Needed**: Everything runs client-side in the browser
- **Responsive UI**: Works on both desktop and mobile devices

## Technologies Used

- [TensorFlow.js](https://www.tensorflow.org/js)
- [face-api.js](https://justadudewhohacks.github.io/face-api.js/)
- HTML5/CSS3/JavaScript
- IndexedDB (for client-side storage)

## How to Use

1. **Collect Face Data**:
   - Click "Collect Face Data"
   - Enter the person's name
   - Position your face in the camera view
   - Press 'S' or click "Save Image" to capture multiple images
   - Press 'Q' or click "Finish" when done

2. **Train Model**:
   - Click "Train Recognition Model"
   - View your dataset summary
   - Click "Start Training" to train the model
   - Wait for training to complete

3. **Recognize Faces**:
   - Click "Recognize Faces"
   - Allow camera access
   - The system will identify recognized faces in real-time
   - Press ESC or click "Stop Recognition" to exit

4. **Manage Data**:
   - Click "Clear All Data" to reset everything
   - Confirm when prompted

## Setup

No installation needed! Just open the HTML file in a modern browser:

1. Chrome or Firefox recommended
2. Allow camera access when prompted
3. First load may take a minute to download models (~5-10MB)

## Browser Support

- Chrome (recommended)
- Firefox
- Edge (latest versions)
- Safari may have limited functionality

## Limitations

- Requires a decent webcam
- Works best with good lighting
- Performance depends on your device hardware
- No cloud/sync functionality (all data stays in your browser)

## Future Improvements

- Add multiple face recognition simultaneously
- Implement face emotion detection
- Add export/import of trained models
- Improve mobile experience

## Contributing

Contributions are welcome! Please open an issue or pull request for any:
- Bug fixes
- Feature requests
- Performance improvements
- Documentation updates

## License

MIT License - Free for personal and commercial use
