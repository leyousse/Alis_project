using System;

using Demoo.Core.Models;
using Demoo.Core.Services;
using Demoo.ViewModels;

using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Navigation;

namespace Demoo.Views
{
    public sealed partial class ImageGalleryDetailPage : Page
    {
        public ImageGalleryDetailPage()
        {
            InitializeComponent();
        }

        private ImageGalleryDetailViewModel ViewModel
        {
            get { return DataContext as ImageGalleryDetailViewModel; }
        }

        protected override void OnNavigatedTo(NavigationEventArgs e)
        {
            base.OnNavigatedTo(e);
            ViewModel.Initialize(e.NavigationMode);
        }

        protected override void OnNavigatingFrom(NavigatingCancelEventArgs e)
        {
            base.OnNavigatingFrom(e);
            if (e.NavigationMode == NavigationMode.Back)
            {
                ViewModel.UpdateConnectedAnimation();
            }
        }
    }
}
