using System;

using Demoo.ViewModels;

namespace Demoo.Views.MasterDetailDetail
{
    public sealed partial class DetailsView
    {
        public DetailsView()
        {
            InitializeComponent();
        }

        public MasterDetailDetailViewModel ViewModel => DataContext as MasterDetailDetailViewModel;
    }
}
