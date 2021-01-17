using System;

using Demoo.ViewModels;

using Windows.UI.Xaml.Controls;

namespace Demoo.Views
{
    public sealed partial class WebViewPage : Page
    {
        public WebViewPage()
        {
            InitializeComponent();

            Loaded += (s, e) => ViewModel.Initialize(webView);
        }

        private WebViewViewModel ViewModel
        {
            get { return DataContext as WebViewViewModel; }
        }
    }
}
