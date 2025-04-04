﻿using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace Project2_20242
{
    class Commons
    {
        /// <summary>
        /// Mở file
        /// </summary>
        /// <returns>Đường dẫn file</returns>
        public string OpenFile(bool isVideo)
        {
            string filter;
            if (isVideo) filter = "Video (*.MP4;*.AVI)|*.MP4;*.AVI";
            else filter = "Images (*.BMP;*.JPG;*.GIF,*.PNG,*.TIFF,*.JPEG)|*.BMP;*.JPG;*.GIF;*.PNG;*.TIFF;*.JPEG";
            OpenFileDialog openFileDialog1 = new OpenFileDialog
            {
                Title = "Browse Text Files",

                CheckFileExists = true,
                CheckPathExists = true,

                DefaultExt = "jpg",
                Filter = filter,
                FilterIndex = 2,
                RestoreDirectory = true,

                ReadOnlyChecked = true,
                ShowReadOnly = true
            };

            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                return openFileDialog1.FileName;
            }
            else return null;
        }

        internal void AddProccesingImageTye(ComboBox comboBox1, List<string> pROCESSING_IMAGE)
        {
            foreach (var item in pROCESSING_IMAGE)
            {
                comboBox1.Items.Add(item);
            }
        }

        internal void ShowImage(PictureBox pictureBox1, Mat image)
        {
            pictureBox1.Image = BitmapConverter.ToBitmap(image);
        }
    }
}