import React from "react";
import { useState } from "react";
import { Input, Button } from "@mui/material";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";

export const App = () => {
    const [session, setSession] = useState<InferenceSession>();
    const [file, setFile] = useState<File>();

    const modelName = "./model.onnx";
    const modelInputShape = [1, 3, 640, 640];

    cv["onRuntimeInitialized"] = async () => {
        // create session
        console.log("Loading YOLOv7 model...");
        const yolov7 = await InferenceSession.create(modelName);

        // warmup main model
        console.log("Warming up model...");
        const tensor = new Tensor(
            "float32",
            new Float32Array(modelInputShape.reduce((a, b) => a * b)),
            modelInputShape
        );
        await yolov7.run({ images: tensor });

        setSession(yolov7);
        console.log("Сессия создана и подготовлена");
    };

    const onFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const target = event.target as HTMLInputElement;
        if (target) setFile(target.files![0]);
    }

    const onFileUpload = async () => {
        if ((file == undefined) || (file?.type != "image/jpeg")) {
            console.log("Загрузите изображение в формате jpg");
            return;
        }
        console.log("Изображение успешно загружено");

        const image = new Image;
        image.src = URL.createObjectURL(file!);

        image.onload = async () => {
            var canvas = document.getElementById('img1') as HTMLCanvasElement;
            var ctx = canvas!.getContext('2d', { willReadFrequently: true });

            canvas.width = image.width;
            canvas.height = image.height;

            var xRatio1 = canvas.width / 640; // set xRatio1
            var yRatio1 = canvas.height / 640; // set yRatio1

            ctx!.drawImage(image!, 0, 0, canvas.width, canvas.height);

            const [modelWidth, modelHeight] = modelInputShape.slice(2);
            const [input, xRatio, yRatio] = preprocessing(image!, modelWidth, modelHeight);
            const tensor = new Tensor("float32", new cv.Mat(input).data32F, modelInputShape); // to ort.Tensor

            const { output } = await session!.run({ images: tensor }); // run session and get output layer

            console.log('output: ', output);

            const boxes = [];

            // looping through output
            for (let r = 0; r < output.size; r += output.dims[1]) {
                const data = output.data.slice(r, r + output.dims[1]); // get rows
                const x0 = Number(data.slice(1)[0]);
                const y0 = Number(data.slice(1)[1]);
                const x1 = Number(data.slice(1)[2]);
                const y1 = Number(data.slice(1)[3]);
                const classId = Number(data.slice(1)[4]);
                const score = Number(data.slice(1)[5]);

                const w = Number(x1) - Number(x0),
                    h = Number(y1) - Number(y0);

                boxes.push({
                    classId: classId,
                    probability: score,
                    bounding: [Number(x0) * Number(xRatio) * Number(xRatio1), Number(y0) * Number(yRatio) * Number(yRatio1), w * Number(xRatio) * Number(xRatio1), h * Number(yRatio) * Number(yRatio1)],
                });
            }

            const classColors = ['#1a2edb', '#1adb39', '#db1a1a']; // Пример цветов
            const classNames = ['bbq', 'chick', 'steck']; // Пример имен классов

            boxes.forEach((box) => {
                const [x1, y1, width, height] = box.bounding;
                const classId = box.classId;
                const className = classNames[classId];
                const color = classColors[classId % classColors.length]; // Использование цвета по id класса

                ctx!.strokeStyle = '#1a2edb'; // тёмно-синий цвет
                ctx!.lineWidth = 5; // толщина линии в 5px
                ctx!.strokeRect(x1, y1, width, height);
                ctx!.fillStyle = color;
                ctx!.font = '20px Arial';
                ctx!.fillText(className, x1, y1 > 20 ? y1 - 10 : y1 + 20);
            });
        }
    }

    const preprocessing = (source: HTMLImageElement, modelWidth: number, modelHeight: number) => {
        const mat = cv.imread(source); // read from img tag
        const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3); // new image matrix
        cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR

        // padding image to [n x n] dim
        const maxSize = Math.max(matC3.rows, matC3.cols); // get max size from width and height
        const xPad = maxSize - matC3.cols, // set xPadding
            xRatio = maxSize / matC3.cols; // set xRatio
        console.log('matC3.cols ', matC3.cols);
        const yPad = maxSize - matC3.rows, // set yPadding
            yRatio = maxSize / matC3.rows; // set yRatio
        console.log('matC3.rows ', matC3.rows);
        console.log('maxSize ', maxSize);
        const matPad = new cv.Mat(); // new mat for padded image
        cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT); // padding black

        const input = cv.blobFromImage(
            matPad,
            1 / 255.0, // normalize
            new cv.Size(modelWidth, modelHeight), // resize to model input size
            new cv.Scalar(0, 0, 0),
            true, // swapRB
            false // crop
        ); // preprocessing image matrix

        // release mat opencv
        mat.delete();
        matC3.delete();
        matPad.delete();

        return [input, xRatio, yRatio];
    };

    return (
        <div>
            <div>
                <Input type="file" onChange={onFileChange} />
            </div>
            <div>
                <Button variant={"contained"} onClick={onFileUpload}>
                    Анализ
                </Button>
                <canvas id="img1"></canvas>
            </div>
        </div>
    )
}