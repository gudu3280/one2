from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB限制
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 初始化模型（全局加载）
model = YOLO(r'D:\flx\YOLOmoxing\app\yolov8n.pt')  # 确保模型路径正确


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # 保存上传文件
    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    # 执行检测
    results = model.predict(upload_path, conf=0.5)
    res_plotted = results[0].plot()[:, :, ::-1]  # BGR转RGB

    # 保存结果
    result_path = os.path.join('static/results', filename)
    cv2.imwrite(result_path, res_plotted)

    return jsonify({
        'original': f'/static/uploads/{filename}',
        'result': f'/static/results/{filename}'
    })


if __name__ == '__main__':
    # 创建必要目录
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)