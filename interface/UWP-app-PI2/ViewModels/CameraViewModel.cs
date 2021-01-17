using System;

using Caliburn.Micro;

using Demoo.EventHandlers;
using Demoo.Helpers;

using Windows.UI.Xaml.Media.Imaging;

namespace Demoo.ViewModels
{
    public class CameraViewModel : Screen
    {
        private BitmapImage _photo;

        public BitmapImage Photo
        {
            get { return _photo; }
            set { Set(ref _photo, value); }
        }

        public void OnPhotoTaken(CameraControlEventArgs args)
        {
            if (!string.IsNullOrEmpty(args.Photo))
            {
                Photo = new BitmapImage(new Uri(args.Photo));
            }
        }
    }
}
