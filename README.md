# Trackman Golfデータの可視化

https://kouichi-c-nakamura.github.io/trackman-golf/

- Trackman GolfはCSV書き出しをサポートしていない。スマホ版でセッションごとにスクリーンショットを複数撮影し、それを生成AIにstitchさせて、表データを再現した。
- Plotlyを使って、動的なデータとして表示。
- 上段は、飛距離（Y軸）、球速（青色）、最高到達点（円の大きさ）
- 下段は、左右打出角（Y軸）、球速（青色）、打出角（円の大きさ）、
