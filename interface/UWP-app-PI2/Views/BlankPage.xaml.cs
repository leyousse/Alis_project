using System;

using Demoo.ViewModels;

using Windows.UI.Xaml.Controls;

namespace Demoo.Views
{
    public sealed partial class BlankPage : Page
    {
        public BlankPage()
        {
            InitializeComponent();
        }

        private BlankViewModel ViewModel
        {
            get { return DataContext as BlankViewModel; }
        }
    }
}
